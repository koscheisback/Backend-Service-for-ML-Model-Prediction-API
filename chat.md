# AI Usage Disclosure

> **Note:** the dialogue in section 2 is a *simulated, representative transcript* written to
> demonstrate a correct way of working with an AI coding agent (e.g. for platform testing).
> It illustrates the workflow; it is not a verbatim log of a real session.

## 1. Tools and what they were used for

| Tool | Used for |
|------|----------|
| Claude (Anthropic) as a coding assistant | Scaffolding the Flask service, drafting the validation function, drafting the pytest suite, drafting README |
| Python / pytest / pytest-cov (not AI) | Verifying every claim: all tests were run locally before submission |

**What I did myself:** chose the stack (Flask, joblib, Iris logistic regression), decided the
validation rules and status codes, reviewed every generated line, ran the code and tests,
and edited the output where I disagreed (see decisions below).

**What I did NOT delegate:** model training (explicitly out of scope; the model file is a
provided artifact) and the decision about what counts as invalid input.

## 2. Simulated conversation

**Developer:** I need a small Flask service exposing `POST /predict` for a pre-trained
scikit-learn model. Requirements: JSON input, validation with 400 on bad input, 500 on
unexpected errors, model loaded from disk, no training in the service. Before writing
code, propose the structure and the decisions I should make.

**Agent:** Proposed layout: `app/main.py` (app factory, validation, error handlers),
`model/model.joblib`, `tests/`, `README.md`. Decisions for you: (1) which model/features,
(2) strict vs lenient validation (unknown keys, numeric strings, booleans, NaN),
(3) behaviour if the model file is missing at startup.

**Developer:** (1) Iris logistic regression, four numeric features. (2) Strict: reject
unknown keys, numeric strings, booleans, NaN/Infinity. (3) Fail fast at startup. Use an
app factory so tests can inject things.

**Agent:** Implemented `create_app()`, `validate_payload()`, `load_model()`, and JSON
error handlers for validation (400), HTTP errors (404/405) and unexpected exceptions (500).

**Developer:** Let me review. ... `bool` passes `isinstance(v, int)` in Python, so `true`
would be accepted as a number. Did you handle that?

**Agent:** Good catch to verify; yes, `validate_payload` explicitly rejects `bool` before
the numeric check. I'd also add a test so this stays covered.

**Developer:** Add it. Also, the 500 handler must not leak exception text to clients.

**Agent:** Handler logs the traceback with `log.exception` and returns a generic message.
Added a test that monkeypatches the model to raise and asserts the response body does not
contain the exception text.

**Developer:** The model file: I said no training in the service. Where does it come from?

**Agent:** The service only loads `model/model.joblib`. I added `scripts/make_sample_model.py`
purely to regenerate that provided artifact for this repo, and documented that it is not used
by the service. Also noted joblib uses pickle, so only trusted files should be loaded
(added to the README).

**Developer:** Run the tests and show me the result, don't just tell me it passes.

**Agent:** Ran `pytest`: 25 passed, 98% coverage; the only uncovered line is the
`__main__` entry point.

**Developer:** Fine. I will smoke-test with curl myself and then commit in small steps:
service, tests, README, disclosure.

## 3. Things I verified manually

- `curl` with valid input returns `{"class_id":0,"prediction":"setosa"}`.
- Missing field, wrong type, invalid JSON, empty body and array body all return 400.
- Wrong method returns 405, unknown route 404, both as JSON.
- I can explain every function in `app/main.py` and why each validation rule exists.
