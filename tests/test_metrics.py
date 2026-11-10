import numpy as np

from speech_commands.metrics import compute_metrics, label_maps


def test_label_maps_round_trip():
    label2id, id2label = label_maps(["no", "up", "yes"])
    assert label2id == {"no": 0, "up": 1, "yes": 2}
    assert id2label == {0: "no", 1: "up", 2: "yes"}


def test_compute_metrics_reports_accuracy_and_macro_f1_from_logits():
    logits = np.array([[2.0, 0.1], [0.3, 1.0], [0.9, 0.2], [0.1, 0.5]])
    labels = np.array([0, 1, 1, 1])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 0.75
    # class 0: P=1/2 R=1 F1=2/3; class 1: P=1 R=2/3 F1=0.8
    assert abs(metrics["macro_f1"] - (2 / 3 + 0.8) / 2) < 1e-9
