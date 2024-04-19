dance_styles = {
    'belly dancing': 0,
    'breakdancing': 1,
    'country line dancing': 2,
    'dancing ballet': 3,
    'dancing charleston': 4,
    'dancing gangnam style': 5,
    'dancing macarena': 6,
    'jumpstyle dancing': 7,
    'mosh pit dancing': 8,
    'robot dancing': 9,
    'salsa dancing': 10,
    'square dancing': 11,
    'swing dancing': 12,
    'tango dancing': 13,
    'tap dancing': 14
}


decoded_dance_styles = {value: key for key, value in dance_styles.items()}


NUM_CLASSES = len(dance_styles)
SIZE = 384
BATCH_SIZE = 64
EPOCHS = 50
SEED = 42
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
