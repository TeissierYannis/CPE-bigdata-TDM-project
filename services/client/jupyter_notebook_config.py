c.NotebookApp.default_url = '/lab/tree/public/index.ipynb'
c.NotebookApp.allow_root = True
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.token = ''
c.NotebookApp.disable_check_xsrf = True
c.NotebookApp.allow_remote_access = True

c.ServerApp.jpserver_extensions = {'jupyterlab': True}

c.NotebookApp.notebook_dir = '/app'

c.ServerApp.allow_terminal = False
c.ServerApp.allow_password_change = False
c.ServerApp.extra_static_paths = ['/app']