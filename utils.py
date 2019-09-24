# Special labels
BLANK_LABEL = '<blank>'
SOS_LABEL = '<sos>'

def init_vocab():

    """Initialize text to ids vocabulary

    Returns:
        Vocabulary
    """

    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    specials = [BLANK_LABEL, SOS_LABEL]

    return {c: i for c, i in zip(specials + [c for c in alphabet],
                                 range(len(alphabet) + len(specials)))}