from fastapi import UploadFile, HTTPException
from pydantic import BaseModel, Field, field_validator, ValidationError
from PIL import Image
import io
from core.config import settings


class FileUploadConfig(BaseModel):
    """Pydantic model for file upload validation rules."""

    filename: str
    content: bytes
    size: int = Field(gt=0)

    @field_validator("filename")
    @classmethod
    def validate_extension(cls, v: str) -> str:
        allowed_exts = set(settings.ALLOWED_EXTENSIONS.split(","))
        ext = v[v.rfind(".") :].lower() if "." in v else ""
        if ext not in allowed_exts:
            raise ValueError(
                f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v > settings.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large. Maximum: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        return v

    @field_validator("content")
    @classmethod
    def validate_content_type(cls, v: bytes) -> bytes:
        allowed_types = set(settings.ALLOWED_IMAGE_TYPES.split(","))

        try:
            # Use PIL to detect image format
            image = Image.open(io.BytesIO(v))
            detected_format = image.format.lower() if image.format else None

            if detected_format not in allowed_types:
                raise ValueError(
                    f"Invalid image format. Detected: {detected_format or 'unknown'}"
                )

            # Verify image integrity
            image.verify()

        except ValueError:
            raise  # Re-raise our validation error
        except Exception:
            raise ValueError("Invalid or corrupted image file")

        return v


async def validate_upload(file: UploadFile) -> bytes:
    """Validate uploaded file using Pydantic model."""
    try:
        content = await file.read()
        await file.seek(0)

        validated = FileUploadConfig(
            filename=file.filename, content=content, size=len(content)
        )

        return validated.content

    except ValidationError as e:
        # Extract only the custom error message
        error_msg = e.errors()[0].get("msg", "Validation failed")
        if error_msg.startswith("Value error, "):
            error_msg = error_msg.replace("Value error, ", "")
        raise HTTPException(status_code=400, detail=error_msg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")

    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")
