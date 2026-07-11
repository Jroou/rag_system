# Chainlit 2.x BaseDataLayer Interface Reference

Chainlit version: **2.11.1**
Primary source: `.venv/lib/python3.12/site-packages/chainlit/`

---

## 1. Abstract methods — required vs optional

All methods in `BaseDataLayer` carry `@abstractmethod`, so Python will refuse to instantiate a
subclass that omits any of them.  That said, several have trivial no-op bodies in the base class
or in the reference implementation (`SQLAlchemyDataLayer`), making them "effectively optional" in
practice.

Source: `.venv/lib/python3.12/site-packages/chainlit/data/base.py`

| Method | Decorator(s) | Notes |
|--------|-------------|-------|
| `get_user` | `@abstractmethod` | Required |
| `create_user` | `@abstractmethod` | Required |
| `delete_feedback` | `@abstractmethod` | Required |
| `upsert_feedback` | `@abstractmethod` | Required |
| `create_element` | `@queue_until_user_message()` + `@abstractmethod` | Required |
| `get_element` | `@abstractmethod` | Required |
| `delete_element` | `@queue_until_user_message()` + `@abstractmethod` | Required |
| `create_step` | `@queue_until_user_message()` + `@abstractmethod` | Required |
| `update_step` | `@queue_until_user_message()` + `@abstractmethod` | Required |
| `delete_step` | `@queue_until_user_message()` + `@abstractmethod` | Required |
| `get_thread_author` | `@abstractmethod` | Base returns `""` — minimal impl OK |
| `delete_thread` | `@abstractmethod` | Required |
| `list_threads` | `@abstractmethod` | Required |
| `get_thread` | `@abstractmethod` | Required |
| `update_thread` | `@abstractmethod` | Also serves as create (upsert) |
| `build_debug_url` | `@abstractmethod` | SQLAlchemy impl returns `""` — trivial OK |
| `close` | `@abstractmethod` | Must at minimum be a no-op async def |
| `get_favorite_steps` | `@abstractmethod` | Required; return `[]` if favorites unsupported |

**Non-abstract convenience method** (do NOT re-implement):

```python
async def set_step_favorite(self, step_dict: "StepDict", favorite: bool) -> "StepDict":
    # Calls update_step internally — base.py lines 117-124
```

---

## 2. Thread CRUD signatures

Source: `base.py` lines 77–103; reference impl: `sql_alchemy.py` lines 195–325.

### `get_thread_author`
```python
async def get_thread_author(self, thread_id: str) -> str
```
Returns the `userIdentifier` string for the thread owner.  Used by ACL checks.

### `get_thread`
```python
async def get_thread(self, thread_id: str) -> Optional["ThreadDict"]
```
Returns a fully-hydrated `ThreadDict` (including `steps` and `elements` lists) or `None`.

### `update_thread` — also serves as create (upsert)
```python
async def update_thread(
    self,
    thread_id: str,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
) -> None
```
There is **no `create_thread`** method.  Chainlit calls `update_thread` the first time a thread
is referenced (triggered by `create_step`), so the implementation must upsert.  The SQLAlchemy
reference does an `INSERT … ON CONFLICT ("id") DO UPDATE` (`sql_alchemy.py` lines 280–297).

### `list_threads`
```python
async def list_threads(
    self,
    pagination: "Pagination",
    filters: "ThreadFilter",
) -> "PaginatedResponse[ThreadDict]"
```
`Pagination` and `ThreadFilter` come from `chainlit.types`.  See section 4 for their shapes.

### `delete_thread`
```python
async def delete_thread(self, thread_id: str) -> None
```
Must cascade: delete feedbacks, elements, steps, then the thread row (`sql_alchemy.py` lines 299–320).

---

## 3. Registering a custom data layer

Source: `callbacks.py` lines 481–491; `data/__init__.py` lines 10–111.

Use the `@cl.data_layer` decorator on a **synchronous factory function** that returns an instance
of your `BaseDataLayer` subclass.  The function must not be async — Chainlit calls it
synchronously during startup.

```python
import chainlit as cl
from my_package.my_store import MyDataLayer

@cl.data_layer
def get_data_layer() -> MyDataLayer:
    return MyDataLayer(...)
```

The decorator stores the callable in `config.code.data_layer`.  `get_data_layer()` (in
`data/__init__.py`) calls it lazily on the first access and caches the result.

**Priority order** when Chainlit resolves the data layer:
1. `@cl.data_layer` factory (highest priority)
2. `DATABASE_URL` environment variable → `ChainlitDataLayer` (SQLAlchemy)
3. `LITERAL_API_KEY` environment variable → `LiteralDataLayer`
4. Nothing → `None` (persistence disabled)

> **Deprecated pattern** — setting `cl.data.data_layer` directly (the module-level variable) still
> works but emits a `DeprecationWarning` (`data/__init__.py` lines 17–23).

---

## 4. Data type shapes

### `PersistedUser`
Source: `user.py` lines 1–43.

```python
@dataclass
class PersistedUser:
    # From User:
    identifier: str           # unique login identifier
    display_name: Optional[str]
    metadata: Dict            # arbitrary JSON
    # From PersistedUserFields:
    id: str                   # UUID primary key
    createdAt: str            # ISO-8601 timestamp string
```

### `ThreadDict`
Source: `types.py` lines 42–51.

```python
class ThreadDict(TypedDict):
    id: str
    createdAt: str
    name: Optional[str]
    userId: Optional[str]          # internal user UUID
    userIdentifier: Optional[str]  # human-readable login identifier
    tags: Optional[List[str]]
    metadata: Optional[Dict]
    steps: List["StepDict"]        # always present, may be empty list
    elements: Optional[List["ElementDict"]]
```

### `StepDict`
Source: `step.py` lines 45–73. All fields `total=False` except the few the code always sets.

```python
class StepDict(TypedDict, total=False):
    name: str
    type: StepType          # "user_message" | "assistant_message" | "tool" | "run" | …
    id: str
    threadId: str
    parentId: Optional[str]
    command: Optional[str]
    modes: Optional[Dict[str, str]]
    streaming: bool
    waitForAnswer: Optional[bool]
    isError: Optional[bool]
    metadata: Dict
    tags: Optional[List[str]]
    input: str
    output: str
    createdAt: Optional[str]
    start: Optional[str]
    end: Optional[str]
    generation: Optional[Dict]
    showInput: Optional[Union[bool, str]]
    defaultOpen: Optional[bool]
    autoCollapse: Optional[bool]
    language: Optional[str]
    icon: Optional[str]
    feedback: Optional[FeedbackDict]
```

### `ElementDict`
Source: `element.py` lines 49–72. All fields `total=False`.

```python
class ElementDict(TypedDict, total=False):
    id: str
    threadId: Optional[str]
    type: ElementType        # "image" | "text" | "pdf" | "audio" | "video" | "file" | "plotly" | "dataframe" | "custom"
    chainlitKey: Optional[str]
    path: Optional[str]
    url: Optional[str]
    objectKey: Optional[str]
    name: str
    display: ElementDisplay  # "inline" | "side" | "page"
    size: Optional[ElementSize]
    language: Optional[str]
    page: Optional[int]
    props: Optional[Dict]
    autoPlay: Optional[bool]
    playerConfig: Optional[dict]
    forId: Optional[str]
    mime: Optional[str]
```

### `Pagination` and `ThreadFilter`
Source: `types.py` lines 54–63.

```python
class Pagination(BaseModel):
    first: int              # page size
    cursor: Optional[str]   # last-seen thread id (cursor-based pagination)

class ThreadFilter(BaseModel):
    feedback: Literal[0, 1] | None = None
    userId: str | None = None
    search: str | None = None
```

### `PaginatedResponse[T]`
Source: `types.py` lines 97–118.

```python
@dataclass
class PaginatedResponse(Generic[T]):
    pageInfo: PageInfo
    data: List[T]
```
```python
@dataclass
class PageInfo:
    hasNextPage: bool
    startCursor: Optional[str]
    endCursor: Optional[str]
```

---

## 5. Gotchas and non-obvious requirements

### No `create_thread` — `update_thread` is the upsert
Chainlit never calls a dedicated `create_thread`.  The first call to `create_step` internally calls
`update_thread(step_dict["threadId"])` with no other arguments.  Your implementation must handle
this "empty" upsert case: insert a row with just the `id` and `createdAt` if the thread does not
exist yet (`sql_alchemy.py` lines 257–298).

### `@queue_until_user_message()` wraps several methods
`create_element`, `delete_element`, `create_step`, `update_step`, and `delete_step` are decorated
with `@queue_until_user_message()` (from `data/utils.py`).  This decorator queues calls made
before the first user message and replays them afterwards.  The decorator is applied at the
**base-class level**, so subclass overrides do NOT inherit it — you must apply it to your own
overrides as well if you want the same queuing behaviour.

```python
from chainlit.data import queue_until_user_message

class MyDataLayer(BaseDataLayer):
    @queue_until_user_message()
    async def create_step(self, step_dict: "StepDict"):
        ...
```

### `list_threads` must handle `userId`-less filters gracefully
The SQLAlchemy reference raises `ValueError` if `filters.userId` is `None`.  Whether that is the
right default for a custom layer is context-dependent, but at minimum the method must not crash
silently.

### `get_thread` must return hydrated `ThreadDict` (with `steps`)
The UI uses `steps` to display message history.  Returning a `ThreadDict` with an absent or empty
`steps` key renders threads as empty.

### `build_debug_url` and `close` are formally required but trivially implemented
The SQLAlchemy layer returns `""` from `build_debug_url` and just disposes the engine in `close`.
For a custom layer that does not need debug links, `return ""` is sufficient.  `close` must exist
and be awaitable even if it does nothing.

### `get_favorite_steps` must return a list
Return `[]` if your storage does not support favorites.  Returning `None` will break iteration in
the UI layer.

### The factory function passed to `@cl.data_layer` must be synchronous
`get_data_layer()` in `data/__init__.py` calls `config.code.data_layer()` synchronously.  If your
initialisation is async (e.g. opening a DB connection), do it lazily inside the async methods or
use `asyncio.get_event_loop().run_until_complete(...)` in the factory (not recommended).  The
cleaner pattern is to accept a connection string and open the pool on first use.
