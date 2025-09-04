# src/common/exceptions.py
class HFUtilsError(Exception):
    """Base exception for HF utilities."""
    pass

class AuthenticationError(HFUtilsError):
    """Raised when authentication fails."""
    pass

class ValidationError(HFUtilsError):
    """Raised when input validation fails."""
    pass

class UploadError(HFUtilsError):
    """Raised when upload operations fail."""
    pass

class DownloadError(HFUtilsError):
    """Raised when download operations fail."""
    pass

class ModelNotFoundError(HFUtilsError):
    """Raised when a model is not found."""
    pass

class DatasetNotFoundError(HFUtilsError):
    """Raised when a dataset is not found."""
    pass

class SpaceNotFoundError(HFUtilsError):
    """Raised when a space is not found."""
    pass

class RepoExistsError(HFUtilsError):
    """Raised when trying to create a repository that already exists."""
    pass