class UpstreamServiceError(RuntimeError):
    """A configured provider could not complete an allowed request."""

    def __init__(self, service: str, message: str) -> None:
        super().__init__(message)
        self.service = service
