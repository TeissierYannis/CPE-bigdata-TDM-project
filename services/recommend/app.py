import os

import pandas as pd
from flask import Flask, jsonify, request
from mysql.connector import pooling
from dotenv import load_dotenv
from collections import namedtuple
import ast
import numpy as np

load_dotenv()

app = Flask(__name__)

cnf = {
    'host': os.getenv("SQL_HOST") or 'localhost',
    'user': os.getenv("SQL_USER") or 'root',
    'port': os.getenv("SQL_PORT") or 3306,
    'password': os.getenv("SQL_PASSWORD") or '',
    'database': os.getenv("SQL_DATABASE") or 'harvest'
}

# Create a connection pool
connection_pool = pooling.MySQLConnectionPool(pool_name="mypool",
                                              pool_size=2,
                                              **cnf)

ImageProperties = namedtuple(
    'ImageProperties', ['name', 'hex_color', 'tags', 'make', 'orientation', 'width', 'height']
)


def get_metadata_from_mariadb_as_imageproperties():
    # Open a connection to the database
    conn = connection_pool.get_connection()
    # Create a cursor
    c = conn.cursor()

    # Retrieve the metadata
    c.execute("""
        SELECT DISTINCT filename, GROUP_CONCAT(CONCAT(mkey, '\t', mvalue) SEPARATOR '\n') AS metadata
        FROM metadata
        WHERE mkey IN ('Make', 'Orientation', 'ImageWidth', 'ImageHeight', 'tags', 'dominant_color')
        GROUP BY filename;
    """)
    metadata = c.fetchall()

    # Close the connection
    conn.close()

    # use the namedtuple ImageProperties to store the metadata
    images = []

    # Loop through the rows of metadata
    for row in metadata:
        try:
            filename, metadata_str = row
            metadata_items = metadata_str.split('\n')
            metadata_dict = {key: value for key, value in (item.split('\t') for item in metadata_items)}

            # Clean dominant colors: convert the string to a list of tuples and extract only the color hex codes
            dominant_colors = ast.literal_eval(metadata_dict.get('dominant_color', '[]'))
            hex_colors = [color[0] for color in dominant_colors]

            # Clean tags: convert the string to a list of strings
            tags = ast.literal_eval(metadata_dict.get('tags', '[]'))

            # Create an ImageProperties object for each row
            image = ImageProperties(
                name=filename,
                hex_color=hex_colors,
                tags=tags,
                make=metadata_dict.get('Make', None),
                orientation=metadata_dict.get('Orientation', None),
                width=metadata_dict.get('ImageWidth', None),
                height=metadata_dict.get('ImageHeight', None)
            )

            # Add the ImageProperties object to the list
            images.append(image)
        except:
            pass

    return images


def clean(data):
    """
    :param data: DataFrame
    :return:
    """
    # 1. first, get only the first hex_color
    # 2. Convert each column to the right type of data
    # 2. Remove rows with missing values

    data['hex_color'] = data['hex_color'].apply(lambda x: x[0] if len(x) > 0 else None)
    data['width'] = pd.to_numeric(data['width'], errors='coerce')
    data['height'] = pd.to_numeric(data['height'], errors='coerce')
    data['orientation'] = pd.to_numeric(data['orientation'], errors='coerce')

    data = data.dropna()

    return data


def preprocess(data):
    # 1. Normalize and scale numerical properties
    # Width and Height: scale to [0, 1] by dividing each value by the maximum value
    # Hex colors: Convert hex colors to RGB values in the range 0-255 and normalize it by dividing by 255 (range [0, 1])

    data['width'] = data['width'] / data['width'].max()
    data['height'] = data['height'] / data['height'].max()

    # Convert hex color to RGB and normalize
    # Remove # from hex color
    data["hex_color"] = data["hex_color"].apply(lambda x: x[1:])
    data["rgb_color"] = data["hex_color"].apply(lambda x: tuple(int(x[i:i + 2], 16) for i in (0, 2, 4)))
    data[["r", "g", "b"]] = pd.DataFrame(data["rgb_color"].tolist(), index=data.index) / 255
    data.drop(["hex_color", "rgb_color"], axis=1, inplace=True)

    # One-hot encode categorical properties
    # 1. tags, make, orientation
    data = pd.concat([data, pd.get_dummies(data["tags"].apply(pd.Series).stack()).sum(level=0)], axis=1)
    data = pd.concat([data, pd.get_dummies(data["make"], prefix="make")], axis=1)
    data["landscape"] = data["orientation"].apply(lambda x: 1 if x == 0 else 0)
    data["portrait"] = data["orientation"].apply(lambda x: 1 if x == 1 else 0)
    data.drop(["tags", "make", "orientation"], axis=1, inplace=True)

    return data


from sklearn.model_selection import train_test_split
import gym
import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics.pairwise import cosine_similarity

@app.route('/recommend', methods=['GET'])
def recommend():
    raw_image_properties = get_metadata_from_mariadb_as_imageproperties()
    # Save raw image properties to a file named raw_image_properties.csv in the shared folder
    pd.DataFrame(raw_image_properties).to_csv('./shared/raw_image_properties.csv', index=False, sep="|")

    # read csv file
    df = pd.read_csv('./shared/raw_image_properties.csv', sep="|")

    # Convert the list of ImageProperties objects to a pandas DataFrame
    df = pd.DataFrame(raw_image_properties)

    clean_image_properties = clean(df)
    processed_image_properties = preprocess(clean_image_properties)

    # Create feature vectors (numpy array)
    feature_vectors = processed_image_properties.drop(["name"], axis=1).values

    # Combine the feature vectors with the image names in a new DataFrame
    data_with_features = processed_image_properties[['name']].copy()
    data_with_features['features'] = list(feature_vectors)

    # Split the dataset into training (70%) and the remaining 30%
    train_data, temp_data = train_test_split(data_with_features, test_size=0.3, random_state=42)

    # Split the remaining data into validation (15%) and test (15%) sets
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    env = ImageRecommenderEnvironment(data_with_features)
    env = DummyVecEnv([lambda: ImageRecommenderEnvironment(data_with_features)])

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_kwargs = dict(
        features_extractor_class=CustomDQNNetwork,
        features_extractor_kwargs=dict(input_dim=input_dim, output_dim=output_dim)
    )

    agent = DQN(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

    total_timesteps = 50000  # Adjust this value based on the problem and computational resources
    agent.learn(total_timesteps=total_timesteps)

    # Save the trained agent
    #agent.save("dqn_agent.pkl")

    # Load the trained agent
    #agent = DQN.load("dqn_agent.pkl")

    # Create validation and test environments
    validation_env = DummyVecEnv([lambda: ImageRecommenderEnvironment(validation_data)])
    test_env = DummyVecEnv([lambda: ImageRecommenderEnvironment(test_data)])

    # Evaluate the agent on the validation and test sets
    num_validation_episodes = 100
    num_test_episodes = 100

    validation_reward = evaluate_agent(agent, validation_env, num_validation_episodes)
    test_reward = evaluate_agent(agent, test_env, num_test_episodes)

    print(f"Validation reward: {validation_reward}")
    print(f"Test reward: {test_reward}")

    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)


    return jsonify(
        {
            'status': 'success',
            'message': 'Recommendation successful',
            'data': {
                'validation_reward': validation_reward,
                'test_reward': test_reward,
                "precision": "NaN",
                "recall": "NaN",
                "f1_score": "NaN"
            }
        }
    )


class ImageRecommenderEnvironment(gym.Env):
    def __init__(self, data_with_features):
        super(ImageRecommenderEnvironment, self).__init__()

        self.data = data_with_features
        self.current_state = None

        # Define action space as the indices of the images
        self.action_space = spaces.Discrete(len(self.data))

        # Define state space as the feature vectors
        feature_vector_length = len(self.data['features'].iloc[0])
        self.observation_space = spaces.Box(low=0, high=1, shape=(feature_vector_length,), dtype=np.float32)

    def reset(self):
        # Reset the environment to an initial state
        self.current_state = self._get_initial_state()
        return self.current_state

    def step(self, action):
        # Execute the given action and observe the next state and reward
        next_state, reward = self._get_next_state_and_reward(action)

        # Check if the episode is done
        done = self._is_done(next_state)

        # Update the current state
        self.current_state = next_state

        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Render the current state of the environment (optional)
        pass

    def _get_initial_state(self):
        # Implement a method to generate an initial state
        pass

    def _get_next_state_and_reward(self, action):
        # Implement a method to get the next state and reward based on the action taken
        pass

    def _is_done(self, next_state):
        # Implement a method to check if the episode is done
        pass


class CustomDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomDQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def reward_function(user_preferences, recommended_image_features):
    similarities = cosine_similarity(user_preferences, recommended_image_features)
    return similarities.mean()

def evaluate_agent(agent, env, num_episodes):
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.predict(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


if __name__ == '__main__':
    app.run(debug=False)
