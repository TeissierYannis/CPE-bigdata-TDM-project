CREATE TABLE image_metadata
(
    id                    SERIAL PRIMARY KEY,
    filename              TEXT UNIQUE,
    Make                  TEXT,
    Model                 TEXT,
    Software              TEXT,
    BitsPerSample         TEXT,
    ImageWidth            TEXT,
    ImageHeight           TEXT,
    ImageDescription      TEXT,
    Orientation           TEXT,
    Copyright             TEXT,
    DateTime              TEXT,
    DateTimeOriginal      TEXT,
    DateTimeDigitized     TEXT,
    SubSecTimeOriginal    TEXT,
    ExposureTime          TEXT,
    FNumber               TEXT,
    ExposureProgram       TEXT,
    ISOSpeedRatings       TEXT,
    SubjectDistance       TEXT,
    ExposureBiasValue     TEXT,
    Flash                 TEXT,
    FlashReturnedLight    TEXT,
    FlashMode             TEXT,
    MeteringMode          TEXT,
    FocalLength           TEXT,
    FocalLengthIn35mm     TEXT,
    Latitude              TEXT,
    LatitudeDegrees       TEXT,
    LatitudeMinutes       TEXT,
    LatitudeSeconds       TEXT,
    LatitudeDirection     TEXT,
    Longitude             TEXT,
    LongitudeDegrees      TEXT,
    LongitudeMinutes      TEXT,
    LongitudeSeconds      TEXT,
    LongitudeDirection    TEXT,
    Altitude              TEXT,
    DOP                   TEXT,
    FocalLengthMin        TEXT,
    FocalLengthMax        TEXT,
    FStopMin              TEXT,
    FStopMax              TEXT,
    LensMake              TEXT,
    LensModel             TEXT,
    FocalPlaneXResolution TEXT,
    FocalPlaneYResolution TEXT,
    tags                  TEXT,
    dominant_color        TEXT
);

CREATE INDEX idx_image_metadata_filename ON image_metadata (filename);

CREATE
OR REPLACE FUNCTION notify_new_metadata()
RETURNS TRIGGER AS $$
DECLARE
payload JSON;
BEGIN
  payload
:= json_build_object(
    'table', TG_TABLE_NAME,
    'action', TG_OP,
    'data', row_to_json(NEW)
  );

  PERFORM
pg_notify('metadata_channel', payload::text);
RETURN NEW;
END;
$$
LANGUAGE plpgsql;

CREATE TRIGGER image_metadata_notify
    AFTER INSERT
    ON image_metadata
    FOR EACH ROW
    EXECUTE FUNCTION notify_new_metadata();