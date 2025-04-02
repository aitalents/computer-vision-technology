from marshmallow import Schema, fields, validate


class RecognitionsSchema(Schema):
    predicted_class = fields.Str(required=True)
    class_probability = fields.Float(required=True)
    contour_probability = fields.Float(required=True)
    x_min = fields.Int(required=True)
    y_min = fields.Int(required=True)
    width = fields.Int(required=True, validate=[validate.Range(
        min=1, error="Value must be greater than 0")])
    height = fields.Int(required=True, validate=[validate.Range(
        min=1, error="Value must be greater than 0")])



class RecognitionResultsSchema(Schema):
    task_id = fields.Str(required=True)
    recognitions = fields.Nested(required=True, nested=RecognitionsSchema)


class RecognitionRequestSchema(Schema):
    task_id = fields.Str(required=True)
    image_url = fields.Str(
        required=True,
        validate=validate.URL(relative=False, require_tld=False)
    )

