from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH
import astroid


class HardcodedHyperparameterChecker(BaseChecker):
    name = "ml-hardcoded-hyperparams"
    priority = -1
    msgs = {
        "W9001": (
            "Hardcoded hyperparameter '%s' detected in model constructor",
            "hardcoded-hyperparameter",
            "Used when a model is constructed with hardcoded hyperparameter values.",
        ),
    }

    def visit_call(self, node):
        model_classes = {"LogisticRegression", "RandomForestClassifier", "XGBClassifier"}

        func_name = (
            node.func.attrname if isinstance(node.func, astroid.Attribute)
            else node.func.name if isinstance(node.func, astroid.Name)
            else None
        )
        if func_name in model_classes:
            for keyword in node.keywords:
                if keyword.value and not keyword.value.as_string().startswith(("param_grid", "config")):
                    self.add_message(
                        "hardcoded-hyperparameter",
                        node=keyword,
                        args=(keyword.arg,)
                    )


class UnseededRandomnessChecker(BaseChecker):
    name = "ml-unseeded-randomness"
    priority = -1
    msgs = {
        "W9002": (
            "Random function used without setting a seed",
            "unseeded-randomness",
            "Random behavior should be reproducible via seeds.",
        ),
    }

    def __init__(self, linter=None):
        super().__init__(linter)
        self.seed_set = False

    def visit_call(self, node):
        if isinstance(node.func, astroid.Attribute):
            if node.func.attrname == "seed":
                self.seed_set = True
            if node.func.attrname in {"randint", "random", "shuffle", "choice"}:
                expr_str = node.func.expr.as_string()
                if "random" in expr_str or "np.random" in expr_str:
                    if not self.seed_set:
                        self.add_message("unseeded-randomness", node=node)

    def leave_module(self, _):
        self.seed_set = False


class TrainingInInferenceChecker(BaseChecker):
    name = "ml-training-in-inference"
    priority = -1
    msgs = {
        "W9003": (
            "Model training detected in a non-training script",
            "training-in-inference",
            "Model training should not happen during inference or in API route handlers.",
        ),
    }

    def visit_call(self, node):
        if isinstance(node.func, astroid.Attribute) and node.func.attrname == "fit":
            filename = node.root().file.lower()
            if "app" in filename or "predict" in filename:
                self.add_message("training-in-inference", node=node)


class SilentDropnaChecker(BaseChecker):
    name = "ml-silent-dropna"
    priority = -1
    msgs = {
        "W9004": (
            "Data dropped via dropna() without logging or warning",
            "silent-dropna",
            "Dropping data silently is a code smell — log it or track it.",
        ),
    }

    def visit_call(self, node):
        if isinstance(node.func, astroid.Attribute) and node.func.attrname == "dropna":
            self.add_message("silent-dropna", node=node)


def register(linter):
    linter.register_checker(HardcodedHyperparameterChecker(linter))
    linter.register_checker(UnseededRandomnessChecker(linter))
    linter.register_checker(TrainingInInferenceChecker(linter))
    linter.register_checker(SilentDropnaChecker(linter))