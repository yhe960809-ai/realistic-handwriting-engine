"""渲染进度回调。

一页 A4 的 v3 排版要数秒，长文档几十秒。调用方需要两件事：知道现在渲到第
几页（用于真实进度条），以及能在超时后让渲染停下来（而不是让线程一直烧
CPU 直到跑完）。

回调通过 contextvar 传递而不是加到 ``RenderConfig`` 上：配置对象会被序列化
进任务记录，塞一个函数进去既不能序列化也会污染契约。contextvar 的作用域就是
当前线程/任务，``run_in_threadpool`` 里设一次即可。

回调抛出的异常会原样向上传播，中断渲染 —— 这就是超时的实现方式：

    with page_progress(lambda done, total: check_deadline()):
        document = render_document(text, config)
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Iterator, Optional

# (已完成页数, 总页数) -> None。抛异常即中断渲染。
PageProgress = Callable[[int, int], None]

_current: contextvars.ContextVar[Optional[PageProgress]] = contextvars.ContextVar(
    "renderer_page_progress", default=None
)


@contextlib.contextmanager
def page_progress(callback: Optional[PageProgress]) -> Iterator[None]:
    """在此上下文内的渲染会逐页回调 ``callback``。"""
    token = _current.set(callback)
    try:
        yield
    finally:
        _current.reset(token)


def report_page(done: int, total: int) -> None:
    """引擎在完成一页后调用。没有注册回调时是空操作。"""
    callback = _current.get()
    if callback is not None:
        callback(done, total)
