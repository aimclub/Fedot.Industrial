"""Call the native prediction protocol or a declared legacy signature once."""

from inspect import signature


def predict_with_horizon(model, history, horizon: int):
    predict = model.predict
    try:
        parameters = signature(predict)
    except (TypeError, ValueError):
        return predict(history, forecast_horizon=horizon)

    try:
        parameters.bind(history, forecast_horizon=horizon)
    except TypeError:
        parameters.bind(history)
        return predict(history)
    return predict(history, forecast_horizon=horizon)
