"""Numbers reported in Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial
Examples" (ICLR 2015, arXiv:1412.6572v3), all on MNIST with eps = 0.25 unless stated otherwise.

Each entry: (value, where in the paper). Error rates and confidences are fractions in [0, 1].
"""

PAPER = {
    "logistic_3v7_clean_error": (0.016, "Fig. 2c"),
    "logistic_3v7_adv_error": (0.99, "Fig. 2d, eps = .25"),
    "softmax_adv_error": (0.999, "Sec. 4"),
    "softmax_adv_confidence": (0.793, "Sec. 4"),
    "maxout_adv_error": (0.894, "Sec. 4"),
    "maxout_adv_confidence": (0.976, "Sec. 4"),
    "maxout_clean_error": (0.0094, "Sec. 6 (dropout, no adversarial training)"),
    "advtrained_clean_error": (0.0084, "Sec. 6 (240 units); 0.782 % with 1600 units"),
    "advtrained_adv_error": (0.179, "Sec. 6 (1600 units)"),
    "advtrained_adv_confidence": (0.814, "Sec. 6"),
    # examples generated with the ORIGINAL model, error measured on the adversarially trained one
    "transfer_original_to_advtrained_error": (0.196, "Sec. 6"),
    # examples generated with the ADVERSARIALLY TRAINED model, error measured on the original one
    "transfer_advtrained_to_original_error": (0.409, "Sec. 6"),
    "noise_sign_adv_error": (0.862, "Sec. 6 control (random +-eps noise)"),
    "noise_sign_adv_confidence": (0.973, "Sec. 6 control"),
    "noise_uniform_adv_error": (0.904, "Sec. 6 control (U(-eps, eps) noise)"),
    "noise_uniform_adv_confidence": (0.978, "Sec. 6 control"),
    "rbf_adv_error": (0.554, "Sec. 7"),
    "rbf_adv_confidence": (0.012, "Sec. 7 (confidence on mistakes)"),
    "rbf_clean_confidence": (0.606, "Sec. 7"),
    "rubbish_softmax_error": (0.598, "App. A"),
    "rubbish_softmax_confidence": (0.708, "App. A"),
    "rubbish_maxout_error": (0.9835, "App. A"),
    "rubbish_maxout_confidence": (0.928, "App. A"),
    "rubbish_rbf_error": (0.0, "App. A"),
    "transfer_maxout_to_softmax_agreement": (0.546, "Sec. 8"),
    "transfer_maxout_to_rbf_agreement": (0.160, "Sec. 8"),
    "transfer_maxout_to_softmax_agreement_both_wrong": (0.846, "Sec. 8"),
    "transfer_maxout_to_rbf_agreement_both_wrong": (0.543, "Sec. 8"),
}
