GENERIC_PROCESSING_FAILURE = (
    "Processing failed after several attempts. Please try again later or upload the file again."
)


class PermanentProcessingError(Exception):
    """The document itself can't be processed, so retrying will never help.

    `message` is written for the end user and is stored in documents.error_message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
