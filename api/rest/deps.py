from fastapi import HTTPException, Request, status


def get_engine(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )
    return engine


def get_replay_store(request: Request):
    store = getattr(request.app.state, "replay_store", None)
    if store is None:
        raise RuntimeError("ReplayStore not initialized")
    return store


def get_secret_provider(request: Request):
    return getattr(request.app.state, "secret_provider", None)
