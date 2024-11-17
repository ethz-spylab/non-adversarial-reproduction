import pydantic
import typing


class PostRecord(pydantic.BaseModel):
    id: str  # id
    author: str  # author name
    was_removed: bool  # post was deleted or removed
    created_utc: int  # timestamp
    title: str  # title
    text: str  # actual text, stripped but case sensitive


class CommentRecord(pydantic.BaseModel):
    id: str  # id
    author: str  # author name
    was_removed: bool  # comment was deleted or removed
    distinguished_as: typing.Optional[
        str
    ]  # distinguished as (typically "moderator" to filter out mod posts)
    created_utc: int  # timestamp
    text: str  # actual text, stripped but case sensitive
    parent_id: str  # direct parent (comment or submission) id
    link_id: str  # parent post id
