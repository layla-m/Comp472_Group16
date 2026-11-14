"""
Neural-network experiment = the protocol shared by every MLP of the project:

1. stratified 80/20 split, feature pipeline fitted on the 80 %, training with early stopping
   -> validation accuracy, best epoch, validation probabilities saved;
2. feature pipeline refitted on 100 %, N_SEEDS models trained for best_epoch epochs,
   test probabilities averaged and saved;
3. submission <experiment name>.csv (image + sentiment).
"""
import time

import numpy as np
import pandas as pd

from .config import N_SEEDS, SEED
from .data import TASKS, load_task
from .features import FeaturePipeline, augmentation_inputs
from .models.mlp import predict_proba
from .models.trainer import MLPTrainer
from .paths import LOGS_DIR
from .store import ProbabilityStore
from .submission import build_submission, predictions, save_submission


class NNExperimentRunner:
    def __init__(self, store=None, n_seeds=N_SEEDS, record_history=True):
        self.store = store or ProbabilityStore()
        self.n_seeds, self.record_history = n_seeds, record_history

    def validate(self, task, cfg):
        """Step 1. Returns the TrainResult of the model trained on the 80 %."""
        data = load_task(task)
        s = data.split
        pipe = FeaturePipeline(cfg).fit(s.X_tr)
        trainer = MLPTrainer(cfg, seed=SEED, record_history=self.record_history)
        res = trainer.fit(pipe.transform(s.X_tr), s.y_tr, data.n_classes, pipe.transform(s.X_va), s.y_va,
                          **augmentation_inputs(cfg, s.X_tr, pipe))
        res.val_proba = predict_proba(res.model, pipe.transform(s.X_va))
        return res

    def final_proba(self, task, cfg, epochs):
        """Step 2. Average test probabilities of N_SEEDS models trained on 100 %."""
        data = load_task(task)
        pipe = FeaturePipeline(cfg).fit(data.X)
        X_full, X_te = pipe.transform(data.X), pipe.transform(data.X_test)
        probs = np.zeros((len(X_te), data.n_classes))
        for s in range(self.n_seeds):
            res = MLPTrainer(cfg, seed=SEED + s).fit(X_full, data.y, data.n_classes, epochs=epochs,
                                                     **augmentation_inputs(cfg, data.X, pipe))
            probs += predict_proba(res.model, X_te)
        return probs / self.n_seeds

    def run_task(self, experiment, task):
        cfg = experiment.config(task)
        t0 = time.time()
        res = self.validate(task, cfg)
        print(f"[{task}] val_acc={res.best_acc:.4f} best_epoch={res.best_epoch} "
              f"({time.time() - t0:.0f}s)", flush=True)
        self.store.save(res.val_proba, experiment.name, task, "val")
        if res.history:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(res.history, columns=["epoch", "train_loss", "train_acc", "val_acc"]).to_csv(
                LOGS_DIR / f"{experiment.name}_{task}_history.csv", index=False)
        P = self.final_proba(task, cfg, res.best_epoch)
        self.store.save(P, experiment.name, task, "test")
        return predictions(load_task(task).test_ids, P.argmax(1)), res.best_acc

    def run(self, experiment):
        print(f"experiment {experiment.name}: {experiment.description}", flush=True)
        out = {task: self.run_task(experiment, task) for task in TASKS}
        (img, a1), (sen, a2) = out["image"], out["sentiment"]
        print(f"Validation: image={a1:.4f} sentiment={a2:.4f} overall={(a1 + a2) / 2:.4f}")
        path = save_submission(build_submission(img, sen), experiment.name)
        print(f"Saved {path}")
        return path
