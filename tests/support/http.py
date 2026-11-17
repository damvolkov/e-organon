"""tests/support/http: an HTTP client of the tests' own — absolute URLs recorded, answers scripted, uploads drained so
the body a driver streamed is kept as the bytes it really sent, not as a promise."""

from typing import TYPE_CHECKING, ClassVar, Self

from e_organon.infra.errors import HTTPError
from e_organon.ports.provider import Isolation, Locality, ProviderSpec, Workload

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping
    from datetime import timedelta


class FakeHTTP:
    """`get` and `upload` answer from `answers` by path tail; every call records its url, headers and bound."""

    spec: ClassVar[ProviderSpec] = ProviderSpec("fake-http", Locality.REMOTE, Workload.IO_BOUND, Isolation.LOOP)

    def __init__(self, answers: Mapping[str, object]) -> None:
        self.answers = dict(answers)
        self.gets: list[str] = []
        self.uploads: list[tuple[str, bytes, timedelta | None, Mapping[str, str] | None]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def health(self) -> bool:
        return True

    async def get(
        self,
        path: str,
        *,
        as_: object = None,
        within: timedelta | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> object:
        self.gets.append(path)
        return self._answer(path)

    async def upload(
        self,
        path: str,
        body: AsyncGenerator[bytes],
        *,
        within: timedelta | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> object:
        sent = b"".join([chunk async for chunk in body])
        self.uploads.append((path, sent, within, headers))
        return self._answer(path)

    async def post(self, path: str, payload: object, **_: object) -> object:
        msg = "the fake posts nothing"
        raise HTTPError(msg)

    async def raw(self, path: str, **_: object) -> bytes:
        msg = "the fake reads no raw bodies"
        raise HTTPError(msg)

    async def delete(self, path: str, **_: object) -> object:
        msg = "the fake deletes nothing"
        raise HTTPError(msg)

    def _answer(self, path: str) -> object:
        for tail, answer in self.answers.items():
            if path.endswith(tail):
                return answer
        msg = f"the fake was not scripted for {path}"
        raise HTTPError(msg)
