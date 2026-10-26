from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class EngineError(Exception):
    """Base class for domain errors. Subclasses set the HTTP status the API maps them to."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_message: str = "Face recognition engine error."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class ModelLoadError(EngineError):
    default_message = "Model weights could not be loaded."


class InvalidImageError(EngineError):
    status_code = status.HTTP_400_BAD_REQUEST
    default_message = "The uploaded file is not a readable image."


class PayloadTooLargeError(EngineError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    default_message = "The uploaded file is too large."


class FaceNotDetectedError(EngineError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    default_message = "No face was detected in the image."


class MultipleFacesError(EngineError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    default_message = "More than one face was detected; enrollment needs exactly one."


class FaceNotFoundError(EngineError):
    status_code = status.HTTP_404_NOT_FOUND
    default_message = "No enrolled person with that id."


class TrackNotFoundError(EngineError):
    status_code = status.HTTP_404_NOT_FOUND
    default_message = "No active track with that id."


class CheckInNotFoundError(EngineError):
    status_code = status.HTTP_404_NOT_FOUND
    default_message = "No check-in with that sequence."


class StreamUnavailableError(EngineError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_message = "The video stream is not running."


async def _handle_engine_error(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, EngineError)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": type(error).__name__, "detail": error.message},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(EngineError, _handle_engine_error)
