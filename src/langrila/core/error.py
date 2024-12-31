class RetryLimitExceededError(Exception):
    def __init__(
        self,
        message: str = "Retry limit exceeded. Please try again or change the request parameters.",
    ) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message
