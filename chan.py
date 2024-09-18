"""
# -*- coding: utf-8 -*-
# @Time    : 2024/04/28 16:45
# @Author  : YuYuKunKun
# @File    : chan.py
"""

import sys
import json
import math
import struct
import asyncio
import time
import traceback
import collections
from copy import copy
import datetime
import itertools
from collections import OrderedDict

from pathlib import Path
from random import choice
from threading import Thread
from typing import (
    List,
    Union,
    Self,
    Literal,
    Optional,
    Tuple,
    final,
    Dict,
    Iterable,
    Any,
    Annotated,
)
from dataclasses import dataclass
from importlib import reload
from enum import Enum
from abc import ABCMeta, abstractmethod, ABC

import matplotlib
import requests
import backtrader as bt
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from termcolor import colored
from backtrader.feeds import GenericCSVData

global_bi = []
global_state = ""

ts2int = lambda timestamp_str: int(
    time.mktime(
        datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timetuple()
    )
)
int2ts = lambda timestamp: time.strftime(
    "%Y-%m-%d %H:%M:%S", time.localtime(int(timestamp))
)


def bs2df(bs: bytes) -> pd.DataFrame:
    size = struct.calcsize(">6d")
    tmp = []
    while bs:
        timestamp, open, high, low, close, vol = struct.unpack(
            ">6d", bs[: struct.calcsize(">6d")]
        )
        tmp.append((int2ts(timestamp), open, high, low, close, vol))
        bs = bs[size:]
    df = pd.DataFrame(
        tmp, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df.index = pd.to_datetime(df.timestamp)
    return df


class Shape(Enum):
    """
    缠论分型
    """

    D = "底分型"
    G = "顶分型"
    S = "上升分型"
    X = "下降分型"
    T = "喇叭口型"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Direction(Enum):
    Up = "向上"
    Down = "向下"
    JumpUp = "缺口向上"
    JumpDown = "缺口向下"

    Left = "左包右"  # 顺序包含
    Right = "右包左"  # 逆序包含

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Freq(Enum):
    # 60 180 300 900 1800 3600 7200 14400 21600 43200 86400 259200
    S1: int = 1
    S3: int = 3
    S5: int = 5
    S12: int = 12
    m1: int = 60 * 1
    m3: int = 60 * 3  # 180
    m5: int = 60 * 5  # 300
    m15: int = 60 * 15  # 900
    m30: int = 60 * 30  # 1800
    H1: int = 60 * 60 * 1  # 3600
    H2: int = 60 * 60 * 2  # 7200
    H4: int = 60 * 60 * 4  # 14400
    H6: int = 60 * 60 * 6  # 21600
    H12: int = 60 * 60 * 12  # 43200
    D1: int = 60 * 60 * 24  # 86400
    D3: int = 60 * 60 * 24 * 3  # 259200


class BSPoint(Enum):
    """买卖点"""

    FS = "一卖"
    FB = "一买"

    SS = "二卖"
    SB = "二买"

    TS = "三卖"
    TB = "三买"


class ChanException(Exception):
    """exception"""

    ...


States = Literal["老阳", "少阴", "老阴", "小阳"]


def _print(*args, **kwords):
    result = []
    for i in args:
        if i in ("小阳", True, Shape.D, "底分型") or "小阳" in str(i):
            result.append(colored(i, "green"))

        elif i in ("老阳", False, Shape.G, "顶分型") or "老阳" in str(i):
            result.append(colored(i, "red"))

        elif i in ("少阴",) or "少阴" in str(i):
            result.append("\33[07m" + colored(i, "yellow"))

        elif i in ("老阴",) or "老阴" in str(i):
            result.append("\33[01m" + colored(i, "blue"))

        elif "PUSH" in str(i):
            result.append(colored(i, "red"))

        elif "POP" in str(i):
            result.append(colored(i, "green"))

        elif "ANALYSIS" in str(i):
            result.append(colored(i, "blue"))

        else:
            result.append(i)
    result = tuple(result)


# print(*result, **kwords)


def dp(*args, **kwords):
    if not 0:
        _print(*args, **kwords)


def bdp(*args, **kwargs):
    if not 1:
        dp(*args, **kwargs)


def ddp(*args, **kwargs):
    if not 1:
        dp(*args, **kwargs)


def zsdp(*args, **kwargs):
    if not 1:
        dp(*args, **kwargs)


class Pillar:
    def __init__(self, high: float, low: float):
        self.low = low
        self.high = high

    def __str__(self):
        return f"Pillar({self.high}, {self.low})"

    def __repr__(self):
        return f"Pillar({self.high}, {self.low})"


def double_relation(left, right) -> Direction:
    """
    两个带有[low, high]对象的所有关系
    """
    # assert hasattr(left, "low")
    # assert hasattr(left, "high")
    # assert hasattr(right, "low")
    # assert hasattr(right, "high")

    relation = None
    assert left is not right, ChanException("相同对象无法比较", left, right)

    if (left.low <= right.low) and (left.high >= right.high):
        relation = Direction.Left  # "左包右" # 顺序

    elif (left.low >= right.low) and (left.high <= right.high):
        relation = Direction.Right  # "右包左" # 逆序

    elif (left.low < right.low) and (left.high < right.high):
        relation = Direction.Up  # "上涨"
        if left.high < right.low:
            relation = Direction.JumpUp  # "跳涨"

    elif (left.low > right.low) and (left.high > right.high):
        relation = Direction.Down  # "下跌"
        if left.low > right.high:
            relation = Direction.JumpDown  # "跳跌"

    return relation


def triple_relation(
    left, mid, right, use_right=False
) -> tuple[Optional[Shape], tuple[Direction, Direction]]:
    """
    三棵缠论k线的所有关系#, 允许逆序包含存在。
    顶分型: 中间高点为三棵最高点。
    底分型: 中间低点为三棵最低点。
    上升分型: 高点从左至右依次升高
    下降分型: 低点从左至右依次降低
    喇叭口型: 高低点从左至右依次更高更低

    """
    if any((left == mid, mid == right, left == right)):
        raise ChanException("相同对象无法比较")

    shape = None
    lm = double_relation(left, mid)
    mr = double_relation(mid, right)
    # lr = double_relation(left, right)

    if lm in (Direction.Up, Direction.JumpUp):
        # 涨
        if mr in (Direction.Up, Direction.JumpUp):
            # 涨
            shape = Shape.S
        if mr in (Direction.Down, Direction.JumpDown):
            # 跌
            shape = Shape.G
        if mr is Direction.Left:
            # 顺序包含
            # print("顺序包含 mr")
            raise ChanException("顺序包含 mr")
        if mr is Direction.Right and use_right:
            # 逆序包含
            shape = Shape.S

    if lm in (Direction.Down, Direction.JumpDown):
        # 跌
        if mr in (Direction.Up, Direction.JumpUp):
            # 涨
            shape = Shape.D
        if mr in (Direction.Down, Direction.JumpDown):
            # 跌
            shape = Shape.X
        if mr is Direction.Left:
            # 顺序包含
            # print("顺序包含 mr")
            raise ChanException("顺序包含 mr")
        if mr is Direction.Right and use_right:
            # 逆序包含
            shape = Shape.X

    if lm is Direction.Left:
        # 顺序包含
        # print("顺序包含 lm")
        raise ChanException("顺序包含 lm")

    if lm is Direction.Right and use_right:
        # 逆序包含
        if mr in (Direction.Up, Direction.JumpUp):
            # 涨
            shape = Shape.D
        if mr in (Direction.Down, Direction.JumpDown):
            # 跌
            shape = Shape.G
        if mr is Direction.Left:
            # 顺序包含
            # print("顺序包含 mr")
            raise ChanException("顺序包含 mr")
        if mr is Direction.Right and use_right:
            # 逆序包含
            shape = Shape.T  # 喇叭口型
    return shape, (lm, mr)


def double_scope(left, right) -> tuple[bool, Optional["Pillar"]]:
    """
    计算重叠范围
    """
    assert left.low < left.high
    assert right.low < right.high

    if left.low < right.high <= left.high:
        # 向下
        return True, Pillar(right.high, left.low)
    if left.low <= right.low < left.high:
        # 向上
        return True, Pillar(left.high, right.low)
    if left.low <= right.low and left.high >= right.high:
        return True, Pillar(right.high, right.low)
    if left.low >= right.low and left.high <= right.high:
        return True, Pillar(left.high, left.low)

    return False, None


def triple_scope(left, mid, right) -> tuple[bool, Optional["Pillar"]]:
    b, p = double_scope(left, mid)
    if b:
        return double_scope(p, right)
    return False, None


def calc_bc(left: List, right: List) -> bool:
    return sum([o.macd for o in left]) > sum([o.macd for o in right])


class Position: ...


class Observer(metaclass=ABCMeta):
    """观察者的基类"""

    CAN = False
    USE_RAW = False
    TIME = 0.05
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    sigals = set()
    sigal = None  # 最新信号
    thread = None

    @abstractmethod
    def update(self, observable: "Observable", **kwords: Any):
        cmd = kwords.get("cmd")
        obj = kwords.get("obj")
        if cmd in (
            Bi.CMD_REMOVE,
            Duan.CMD_REMOVE,
            ZhongShu.CMD_REMOVE,
            FeatureSequence.CMD_REMOVE,
        ):
            assert self._removed is False, self
            assert self._appended is True, self
            self._removed = True
            self._appended = False
            del observable

        if cmd in (
            Bi.CMD_APPEND,
            Duan.CMD_APPEND,
            ZhongShu.CMD_APPEND,
            FeatureSequence.CMD_APPEND,
        ):
            assert self._appended is False, self
            self._appended = True
            self._removed = False
        if cmd in (
            Bi.CMD_MODIFY,
            Duan.CMD_MODIFY,
            ZhongShu.CMD_MODIFY,
            FeatureSequence.CMD_MODIFY,
        ):
            assert self._appended is True, self
            assert self._removed is False, self

    @classmethod
    def plot_bar(cls, bar: Union["NewBar", "RawBar"]):
        if not cls.CAN:
            return
        message = {
            "type": "realtime",
            "timestamp": bar.dt.isoformat(),
            "open": bar.open,
            "close": bar.close,
            "high": bar.high,
            "low": bar.low,
            "volume": bar.volume,
        }
        future = asyncio.run_coroutine_threadsafe(
            Observer.queue.put(message), Observer.loop
        )
        try:
            future.result()  # 确保任务已添加到队列
        except Exception as e:
            print(f"Error adding task to queue: {e}")

    @classmethod
    def plot_bsp(cls, _type, bar: "NewBar", bsp: BSPoint, real: bool = True):
        if cls.CAN:
            offset = NewBar.OBJS[-1].index - bar.index
            points = [
                {"time": int(bar.dt.timestamp()), "price": bar.speck},
            ]
            if (_type, bar, bsp) in Observer.sigals and real:
                Observer.sigal = None
                return
            ##print("plot_bsp", bar, bsp, RawBar.OBJS[-1].dt)

            if bar.shape is Shape.G:
                options = {
                    "shape": "arrow_down",
                    "text": f"{_type}{bsp.value}{offset}",
                }
                properties = {"title": f"{_type}{bsp.name}"}
            else:
                options = {
                    "shape": "arrow_up",
                    "text": f"{_type}{bsp.value}{offset}",
                }
                properties = {"title": f"{_type}{bsp.name}"}
            if not real:
                properties["color"] = "#FFFA50"
            message = {
                "type": "shape",
                "cmd": ZhongShu.CMD_APPEND if real else ZhongShu.CMD_MODIFY,
                "name": options["shape"],
                "id": bar.shape_id + _type,
                "points": points,
                "options": options,
                "properties": properties,
            }
            future = asyncio.run_coroutine_threadsafe(
                Observer.queue.put(message), Observer.loop
            )

            try:
                future.result()  # 确保任务已添加到队列
            except Exception as e:
                print(f"Error adding task to queue: {e}")
        Observer.sigals.add((_type, bar, bsp))
        Observer.sigal = (_type, bar, bsp)


class Observable(object):
    """被观察者的基类"""

    __slots__ = ("__observers", "_removed", "_appended")

    def __init__(self):
        self.__observers = []
        self._removed = False
        self._appended = False

    def none_observer(self):
        self.__observers = None

    def new_observer(self):
        self.__observers = []

    # 清空观察者
    def clear_observer(self):
        self.__observers.clear()

    # 添加观察者
    def attach_observer(self, observer: Observer):
        self.__observers.append(observer)

    # 删除观察者
    def detach_observer(self, observer: Observer):
        self.__observers.remove(observer)

    # 内容或状态变化时通知所有的观察者
    def notify_observer(self, **kwords):
        assert kwords.get("obj") is not None
        assert kwords.get("obj") is self
        if Observer.CAN:
            if self.__observers is None:
                # print("警告 观察者 为 None", kwords)
                return
            name = self.__class__.__name__
            if name in ("Bi", "Duan", "ZhongShu", "FeatureSequence"):
                NewBar.OBJS  # andprint(
                #     "\n", name, NewBar.OBJS[-1].dt, kwords, self.__observers
                # )
            for o in self.__observers:
                o.update(self, **kwords)


class TVShapeID(object):
    """
    charting_library shape ID 管理
    """

    IDS = set()
    SHAPES = {
        "emoji",
        "triangle",
        "curve",
        "circle",
        "ellipse",
        "path",
        "polyline",
        "text",
        "icon",
        "extended",
        "anchored_text",
        "anchored_note",
        "note",
        "signpost",
        "double_curve",
        "arc",
        "sticker",
        "arrow_up",
        "arrow_down",
        "arrow_left",
        "arrow_right",
        "price_label",
        "price_note",
        "arrow_marker",
        "flag",
        "vertical_line",
        "horizontal_line",
        "cross_line",
        "horizontal_ray",
        "trend_line",
        "info_line",
        "trend_angle",
        "arrow",
        "ray",
        "parallel_channel",
        "disjoint_angle",
        "flat_bottom",
        "anchored_vwap",
        "pitchfork",
        "schiff_pitchfork_modified",
        "schiff_pitchfork",
        "balloon",
        "comment",
        "inside_pitchfork",
        "pitchfan",
        "gannbox",
        "gannbox_square",
        "gannbox_fixed",
        "gannbox_fan",
        "fib_retracement",
        "fib_trend_ext",
        "fib_speed_resist_fan",
        "fib_timezone",
        "fib_trend_time",
        "fib_circles",
        "fib_spiral",
        "fib_speed_resist_arcs",
        "fib_channel",
        "xabcd_pattern",
        "cypher_pattern",
        "abcd_pattern",
        "callout",
        "triangle_pattern",
        "3divers_pattern",
        "head_and_shoulders",
        "fib_wedge",
        "elliott_impulse_wave",
        "elliott_triangle_wave",
        "elliott_triple_combo",
        "elliott_correction",
        "elliott_double_combo",
        "cyclic_lines",
        "time_cycles",
        "sine_line",
        "long_position",
        "short_position",
        "forecast",
        "date_range",
        "price_range",
        "date_and_price_range",
        "bars_pattern",
        "ghost_feed",
        "projection",
        "rectangle",
        "rotated_rectangle",
        "brush",
        "highlighter",
        "regression_trend",
        "fixed_range_volume_profile",
    }

    __slots__ = "__shape_id"

    def __init__(self):
        super().__init__()
        s = TVShapeID.get(12)
        while s in TVShapeID.IDS:
            s = TVShapeID.get(12)
        TVShapeID.IDS.add(s)
        self.__shape_id: str = s

    def __del__(self):
        TVShapeID.IDS.remove(self.__shape_id)
        self.__shape_id = None

    @property
    def shape_id(self) -> str:
        return self.__shape_id

    @shape_id.setter
    def shape_id(self, value: str):
        TVShapeID.IDS.remove(self.__shape_id)
        self.__shape_id = value
        TVShapeID.IDS.add(value)

    @staticmethod
    def get(size: int):
        return "".join(
            [
                choice("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                for _ in range(size)
            ]
        )

    @classmethod
    def intervalsVisibilities(cls):
        return (
            {
                "ticks": True,
                "seconds": True,
                "secondsFrom": 1,
                "secondsTo": 59,
                "minutes": True,
                "minutesFrom": 1,
                "minutesTo": 59,
                "hours": True,
                "hoursFrom": 1,
                "hoursTo": 24,
                "days": True,
                "daysFrom": 1,
                "daysTo": 366,
                "weeks": True,
                "weeksFrom": 1,
                "weeksTo": 52,
                "months": True,
                "monthsFrom": 1,
                "monthsTo": 12,
                "ranges": True,
            },
        )


class BaseChanObject(Observable):
    """ """

    __slots__ = "cache", "elements", "__done", "index", "pre", "__shape_id", "__pillar"
    FAST = 12
    SLOW = 26
    SIGNAL = 9

    CMD_DONE = "done"

    def __init__(self, index=0):
        super().__init__()
        self.__shape_id = TVShapeID()
        self.__pillar = Pillar(0.0, 0.0)
        self.cache = dict()
        self.elements = []
        self.__done = False
        self.index = index
        self.pre: Optional[Union["RawBar", "NewBar", "Bi", "Duan"]] = None

    @property
    def high(self) -> float:
        return self.__pillar.high

    @high.setter
    def high(self, value: float):
        self.__pillar.high = value

    @property
    def low(self) -> float:
        return self.__pillar.low

    @low.setter
    def low(self, value: float):
        self.__pillar.low = value

    @property
    def done(self) -> bool:
        return self.__done

    @done.setter
    def done(self, value: bool):
        self.__done = value
        self.notify_observer(
            cmd=f"{self.__class__.__name__}_{BaseChanObject.CMD_DONE}", obj=self
        )

    @property
    def macd(self) -> float:
        return sum(abs(bar.macd) for bar in self.elements)

    @property
    def shape_id(self) -> str:
        return self.__shape_id.shape_id

    @shape_id.setter
    def shape_id(self, str6id: str):
        self.__shape_id.shape_id = str6id

    @classmethod
    def last(cls) -> Optional[Union["RawBar", "NewBar", "Bi", "Duan"]]:
        return cls.OBJS[-1] if cls.OBJS else None


class RawBar(BaseChanObject, Observer):
    OBJS: List["RawBar"] = []
    PATCHES: Dict[int, Pillar] = dict()

    CMD_APPEND = "append"

    __slots__ = (
        "open",
        "close",
        "volume",
        "dts",
        "lv",
        "start_include",
        "belong_include",
        "shape",
    )

    def __init__(
        self,
        dt: datetime.datetime,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        i: int,
    ):
        if RawBar.OBJS:
            i = RawBar.last().index + 1
        super().__init__(index=i)

        self.dt = dt
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        if pillar := RawBar.PATCHES.get(int(dt.timestamp())):
            self.high = pillar.high
            self.low = pillar.low

        self.dts = [
            self.dt,
        ]
        self.lv = (
            self.volume
        )  # 最新成交量，用于Tick或频繁获取最新数据时对于相同时间戳的成交量计算“真实成交量可靠性”
        self.start_include: bool = False  # 起始包含位
        self.belong_include: int = -1  # 所属包含
        self.shape: Optional[Shape] = None

        self.elements = None
        RawBar.OBJS.append(self)
        self.update(self, cmd=RawBar.CMD_APPEND)
        self.none_observer()

    def __eq__(self, other):
        if (
            isinstance(other, RawBar)
            and self.index == other.index
            and self.open == other.open
            and self.high == other.high
            and self.low == other.low
            and self.close == other.close
            and self.volume == other.volume
            and self.dt.timestamp() == other.dt.timestamp()
        ):
            return True
        return False

    def __str__(self):
        return f"{self.__class__.__name__}({self.dt}, {self.high}, {self.low}, index={self.index})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dt}, {self.high}, {self.low}, index={self.index})"

    def update(self, observer: "Observer", **kwords: Any):
        if not Observer.CAN:
            return

        if not Observer.USE_RAW:
            return

        cmd = kwords.get("cmd")
        if cmd in (RawBar.CMD_APPEND,):
            Observer.plot_bar(self)

    def __bytes__(self):
        return struct.pack(
            ">6d",
            int(self.dt.timestamp()),
            round(self.open, 8),
            round(self.high, 8),
            round(self.low, 8),
            round(self.close, 8),
            round(self.volume, 8),
        )

    @classmethod
    def from_bytes(cls, buf: bytes):
        timestamp, open, high, low, close, vol = struct.unpack(
            ">6d", buf[: struct.calcsize(">6d")]
        )
        return cls(
            dt=datetime.datetime.fromtimestamp(timestamp),
            o=open,
            h=high,
            l=low,
            c=close,
            v=vol,
            i=0,
        )

    def to_new_bar(self, pre: Optional["NewBar"]) -> "NewBar":
        return NewBar(
            dt=self.dt,
            high=self.high,
            low=self.low,
            elements=[
                self,
            ],
            pre=pre,
        )

    @property
    def macd(self) -> float:
        return self.cache[
            f"macd_{BaseChanObject.FAST}_{BaseChanObject.SLOW}_{BaseChanObject.SIGNAL}"
        ]

    @property
    def ampl(self) -> float:
        """涨跌幅"""
        return (self.open - self.close) / self.open

    @property
    def direction(self) -> Direction:
        return Direction.Up if self.open < self.close else Direction.Down


class NewBar(BaseChanObject, Observer):
    OBJS: List["NewBar"] = []

    CMD_APPEND = "append"

    __slots__ = (
        "__shape",
        "relation",
        "jump",
        "speck",
        "dt",
        "direction",
        "bsp",
    )

    def __init__(
        self,
        dt: datetime.datetime,
        high: float,
        low: float,
        elements: List[RawBar],
        pre: Optional["NewBar"] = None,
    ):
        super().__init__()
        self.__shape: Optional[Shape] = None
        self.relation: Optional[Direction] = None  # 与前一个关系
        self.jump: bool = False  # 与前一个是否是跳空
        self.speck: Optional[float] = None  # 分型高低点

        self.dt = dt
        self.high = high
        self.low = low
        self.elements: List[RawBar] = elements
        # self.pre = pre
        self.direction = self.elements[
            0
        ].direction  # if self.elements else Direction.Up
        if pre is not None:
            relation = double_relation(pre, self)
            assert relation not in (Direction.Left, Direction.Right)
            self.index = pre.index + 1
            if relation in (Direction.JumpUp, Direction.JumpDown):
                self.jump = True
            self.relation = relation
            self.direction = (
                Direction.Up
                if relation in (Direction.JumpUp, Direction.Up)
                else Direction.Down
            )
        self.bsp: List[BSPoint] = []
        NewBar.OBJS.append(self)
        self.update(self, cmd=NewBar.CMD_APPEND)
        self.none_observer()

    def update(self, observable: "Observable", **kwords: Any):
        if not Observer.CAN:
            return

        if Observer.USE_RAW:
            return

        cmd = kwords.get("cmd")
        if cmd in (NewBar.CMD_APPEND,):
            Observer.plot_bar(self)

    @classmethod
    def get_last_fx(cls) -> Optional["FenXing"]:
        try:
            left, mid, right = NewBar.OBJS[-3:]
        except ValueError:
            return

        left, mid, right = NewBar.OBJS[-3:]
        shape, relations = triple_relation(left, mid, right)
        mid.shape = shape

        if shape is Shape.G:
            mid.speck = mid.high
            right.speck = right.low

        if shape is Shape.D:
            mid.speck = mid.low
            right.speck = right.high

        if shape is Shape.S:
            right.speck = right.high
            right.shape = Shape.S
            mid.speck = mid.high

        if shape is Shape.X:
            right.speck = right.low
            right.shape = Shape.X
            mid.speck = mid.low

        return FenXing(left, mid, right)

    def __str__(self):
        return f"{self.__class__.__name__}({self.index}, {self.dt}, {self.high}, {self.low}, {self.shape}, {len(self.elements)})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.index}, {self.dt}, {self.high}, {self.low}, {self.shape}, {len(self.elements)})"

    def _to_raw_bar(self) -> RawBar:
        return RawBar(
            dt=self.dt,
            o=self.open,
            h=self.high,
            l=self.low,
            c=self.close,
            v=self.volume,
            i=self.index,
        )

    def merge(self, next_raw_bar: RawBar) -> Optional["NewBar"]:
        """
        去除包含关系
        :param next_raw_bar :
        :return: 存在包含关系返回 None, 否则返回下一个 NewBar
        """
        assert next_raw_bar.index - 1 == self.elements[-1].index, (
            next_raw_bar,
            self.elements[-1].index,
            self,
        )
        relation = double_relation(self, next_raw_bar)
        if relation in (Direction.Left, Direction.Right):
            # 合并
            if self.direction is Direction.Up:
                self.high = max(self.high, next_raw_bar.high)
                self.low = max(self.low, next_raw_bar.low)
            else:
                self.high = min(self.high, next_raw_bar.high)
                self.low = min(self.low, next_raw_bar.low)

            assert next_raw_bar.index - 1 == self.elements[-1].index
            self.update(self, cmd=NewBar.CMD_APPEND)
            self.elements.append(next_raw_bar)
            return None

        self.done = True
        return next_raw_bar.to_new_bar(self)

    @property
    def shape(self) -> Optional[Shape]:
        return self.__shape

    @shape.setter
    def shape(self, shape: Shape):
        self.__shape = shape
        if shape is None:
            self.speck = None
        if shape is Shape.G:
            self.speck = self.high
        if shape is Shape.S:
            self.speck = self.high
        if shape is Shape.D:
            self.speck = self.low
        if shape is Shape.X:
            self.speck = self.low

    @property
    def volume(self) -> float:
        """
        :return: 总计成交量
        """
        return sum([raw.volume for raw in self.elements])

    @property
    def open(self) -> float:
        return self.high if self.direction == Direction.Down else self.low

    @property
    def close(self) -> float:
        return self.low if self.direction == Direction.Down else self.high


class FenXing(BaseChanObject):
    __slots__ = "left", "mid", "right", "__shape", "__speck"
    OBJS: List["FenXing"] = []

    def __init__(self, left: NewBar, mid: NewBar, right: NewBar, index: int = 0):
        super().__init__()
        self.left = left
        self.mid = mid
        self.right = right
        self.index = index

        self.__shape = mid.shape
        self.__speck = mid.speck
        self.elements = [left, mid, right]

    def next_new_bar(self, next_new_bar: NewBar) -> None:
        assert next_new_bar.index - 1 == self.right.index
        self.done = True

    @property
    def dt(self) -> datetime.datetime:
        return self.mid.dt

    @property
    def shape(self) -> Shape:
        return self.__shape

    @property
    def speck(self) -> float:
        return self.__speck

    @property
    def high(self) -> float:
        return max(self.left.high, self.mid.high)

    @property
    def low(self) -> float:
        return min(self.left.low, self.mid.low)

    def __str__(self):
        return f"FenXing({self.shape}, {self.speck}, {self.dt})"

    def __repr__(self):
        return f"FenXing({self.shape}, {self.speck}, {self.dt})"

    @staticmethod
    def append(fxs, fx):
        if fxs and fxs[-1].shape is fx.shape:
            raise ChanException("分型相同无法添加", fxs[-1], fx)
        i = 0
        if fxs:
            i = fxs[-1].index + 1
        fx.index = i
        fxs.append(fx)

    @staticmethod
    def pop(fxs, fx):
        if fxs and fxs[-1] is not fx:
            raise ChanException("分型相同无法删除", fxs[-1], fx)
        return fxs.pop()


class Bi(BaseChanObject, Observer):
    """
    缠论笔
    """

    OBJS: List["Bi"] = []
    FAKE: "Bi" = None  # 最新未完成笔

    BI_LENGTH = 5  # 成BI最低长度
    BI_JUMP = True  # 跳空是否是一个NewBar
    BI_EQUAL = True  # True: 一笔终点存在多个终点时，取最后一个, False: 用max/min时只会取第一个值，会有这个情况 当首个出现时 小于[BI_LENGTH]而后个则大于[BI_LENGTH]但max/min函数不会取后一个. 例子: bitstamp btcusd 30m [2024-06-03 17:00]至[2024-06-05 01:00] 中 [NewBar(63, 2024-06-03 22:30:00, 69318.0, 68553.0, D, 2), NewBar(94, 2024-06-04 17:30:00, 68768.0, 68553.0, D, 1)]
    BI_FENGXING = False  # True: 一笔起始分型高低包含整支笔对象则不成笔, False: 只判断分型中间数据是否包含
    CMD_APPEND = "bi_append"
    CMD_MODIFY = "bi_modify"
    CMD_REMOVE = "bi_remove"

    __slots__ = "direction", "__start", "__end"

    def __init__(
        self,
        pre: Optional["Self"],
        start: FenXing,
        end: Union[FenXing, NewBar],
        elements: Optional[List[NewBar]],
    ):
        super().__init__()
        if start.shape is Shape.G:
            self.direction = Direction.Down
            self.high = start.speck
            self.low = end.low
        elif start.shape is Shape.D:
            self.direction = Direction.Up
            self.high = end.high
            self.low = start.speck
        else:
            raise ChanException(start.shape, end.shape)
        for i in range(1, len(self.elements)):
            assert self.elements[i - 1].index + 1 == self.elements[i].index, (
                self.elements[i - 1].index,
                self.elements[i].index,
            )
        if pre is not None:
            assert pre.end is start, (pre.end, start)
            self.index = pre.index + 1
        self.pre = pre
        self.__start = start
        self.__end = end
        self.elements = elements
        """if Bi.OBJS:
            last = Bi.OBJS[-1]
            assert last.elements[-1] is elements[0], (
                last.elements[-1],
                elements[0],
            )"""

        self.attach_observer(self)  # 自我观察

    def __str__(self):
        return f"Bi({self.direction}, {colored(self.start.dt, 'green')}, {self.start.speck}, {colored(self.end.dt, 'green')}, {self.end.speck}, {self.index}, {self.elements[-1]}, fake: {self is Bi.FAKE})"

    def __repr__(self):
        return f"Bi({self.direction}, {colored(self.start.dt, 'green')}, {self.start.speck}, {colored(self.end.dt, 'green')}, {self.end.speck}, {self.index}, {self.elements[-1]}, fake: {self is Bi.FAKE})"

    def update(self, observable: "Observable", **kwords: Any):
        # 实现 自我观察
        # return
        cmd = kwords.get("cmd")
        points = [
            {"time": int(self.start.dt.timestamp()), "price": self.start.speck},
            {"time": int(self.elements[-1].dt.timestamp()), "price": self.end.speck},
        ]
        options = {
            "shape": "trend_line",
            # "showInObjectsTree": True,
            # "disableSave": False,
            # "disableSelection": True,
            # "disableUndo": False,
            # "filled": True,
            # "lock": False,
            "text": "bi",
        }
        properties = {
            "linecolor": "#CC62FF",
            "linewidth": 2,
            "title": f"Bi-{self.index}",
        }

        message = {
            "type": "shape",
            "cmd": cmd,
            "name": "trend_line",
            "id": self.shape_id,
            "points": points,
            "options": options,
            "properties": properties,
        }

        if cmd in (Bi.CMD_APPEND, Bi.CMD_REMOVE, Bi.CMD_MODIFY):
            # 后端实现 增 删 改
            future = asyncio.run_coroutine_threadsafe(
                Observer.queue.put(message), Observer.loop
            )
            try:
                future.result()  # 确保任务已添加到队列
            except Exception as e:
                print(f"Error adding task to queue: {e}")
        super().update(observable, **kwords)

    @property
    def length(self) -> int:
        return Bi.calc_length(self.elements)

    @property
    def start(self) -> FenXing:
        return self.__start

    @start.setter
    def start(self, start: FenXing):
        raise ChanException()

    @property
    def end(self) -> FenXing:
        return self.__end

    @end.setter
    def end(self, end: Union[FenXing, NewBar]):
        old = self.__end
        self.__end = end
        tag = True
        if self.direction is Direction.Down:
            if old.low == end.low:
                tag = False
            self.low = min(self.low, end.low)
        if self.direction is Direction.Up:
            if old.high == end.high:
                tag = False
            self.high = max(self.high, end.high)
        if tag:
            self.notify_observer(cmd=Bi.CMD_MODIFY, obj=self)

    @property
    def real_high(self) -> Optional[NewBar]:
        if not self.elements:
            return None
        if Bi.BI_EQUAL:
            high = [self.elements[0]]
            for bar in self.elements[1:]:
                if bar.high >= high[-1].high:
                    if bar.high > high[-1].high:
                        high.clear()
                    high.append(bar)
            if len(high) > 1:
                dp("", high)
            return high[-1]

        return max(self.elements, key=lambda x: x.high)

    @property
    def real_low(self) -> Optional[NewBar]:
        if not self.elements:
            return None
        if Bi.BI_EQUAL:
            low = [self.elements[0]]
            for bar in self.elements[1:]:
                if bar.low <= low[-1].low:
                    if bar.low < low[-1].low:
                        low.clear()
                    low.append(bar)
            if len(low) > 1:
                dp("", low)
            return low[-1]
        return min(self.elements, key=lambda x: x.low)

    @property
    def relation(self) -> bool:
        if Bi.BI_FENGXING:
            start = self.start
        else:
            start = self.start.mid

        if self.direction is Direction.Down:
            return double_relation(start, self.end) in (
                Direction.Down,
                Direction.JumpDown,
            )
        return double_relation(start, self.end) in (Direction.Up, Direction.JumpUp)

    @staticmethod
    def calc_length(elements) -> int:
        size = 1
        # elements = self.elements
        for i in range(1, len(elements)):
            left = elements[i - 1]
            right = elements[i]
            assert left.index + 1 == right.index, (
                left.index,
                right.index,
            )
            relation = double_relation(left, right)
            if Bi.BI_JUMP and relation in (Direction.JumpUp, Direction.JumpDown):
                size += 1
            size += 1
        if not Bi.BI_JUMP:
            assert size == len(elements)
        if Bi.BI_JUMP:
            return size
        return len(elements)

    def get_bsp(self) -> List[NewBar]:
        bsp = []
        for bar in self.elements:
            if bar.bsp:
                bsp.append(bar)
        return bsp

    def check(self) -> bool:
        if len(self.elements) >= 5:
            assert self.start.mid is self.elements[0]
            assert self.end.mid is self.elements[-1]
            if (
                self.direction is Direction.Down
                and self.start.mid is self.real_high
                and self.end.mid is self.real_low
            ):
                return True
            if (
                self.direction is Direction.Up
                and self.start.mid is self.real_low
                and self.end.mid is self.real_high
            ):
                return True
        return False

    @classmethod
    def calc_fake(cls):
        last = cls.FAKE
        if last is not None:
            last.notify_observer(cmd=Bi.CMD_REMOVE, obj=last)

        if FenXing.OBJS:
            start = FenXing.OBJS[-1]
            elememts = NewBar.OBJS[start.mid.index :]
            low = min(elememts, key=lambda x: x.low)
            high = max(elememts, key=lambda x: x.high)
            pre = cls.last()
            if start.shape is Shape.G:
                bi = Bi(pre, start, low, elememts)
            else:
                bi = Bi(pre, start, high, elememts)
            cls.FAKE = bi
            bi.notify_observer(cmd=Bi.CMD_APPEND, obj=bi)

    @staticmethod
    def append(bis, bi, _from):
        if bis and bis[-1].end is not bi.start:
            raise ChanException("笔连续性错误")
        i = 0
        pre = None
        if bis:
            i = bis[-1].index + 1
            pre = bis[-1]
        bi.index = i
        bi.pre = pre
        bi.done = True
        bis.append(bi)
        if _from == "analyzer":
            with open("bi.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"{bi.direction},{bi.start.dt.strftime('%Y%m%d')},{bi.start.speck},"
                    f"{bi.end.dt.strftime('%Y%m%d')},{bi.end.speck}\n"
                )
            global global_bi
            global_bi.append(
                f"{bi.direction},{bi.start.dt.strftime('%Y%m%d')},{bi.start.speck},"
                f"{bi.end.dt.strftime('%Y%m%d')},{bi.end.speck}\n"
            )
            bi.notify_observer(cmd=Bi.CMD_APPEND, obj=bi)
            ZhongShu.analyzer_push(bi, ZhongShu.BI_OBJS, Bi.OBJS, 0, _from)
            Duan.analyzer_append(bi, Duan.OBJS, 0, _from)

    @staticmethod
    def pop(bis, fx, _from):
        if bis:
            if bis[-1].end is fx:
                bi = bis.pop()
                bi.done = False
                if _from == "analyzer":
                    bi.notify_observer(cmd=Bi.CMD_REMOVE, obj=bi)
                    ZhongShu.analyzer_pop(bi, ZhongShu.BI_OBJS, 0, _from)
                    Duan.analyzer_pop(bi, Duan.OBJS, 0, _from)
                return bi
            else:
                raise ChanException("最后一笔终点错误", fx, bis[-1].end)

    @staticmethod
    def analyzer(
        fx: FenXing,
        fxs: List[FenXing],
        bis: List["Bi"],
        cklines: List[NewBar],
        _from: str = "analyzer",
        level=0,
    ):
        bdp(level, fx.mid, _from)
        last = fxs[-1] if fxs else None
        left, mid, right = fx.left, fx.mid, fx.right
        if Bi.FAKE:
            Bi.FAKE.notify_observer(cmd=Bi.CMD_REMOVE, obj=Bi.FAKE)
            Bi.FAKE = None

        if last is None:
            if mid.shape in (Shape.G, Shape.D):
                fxs.append(fx)
            return

        if last.mid.dt > fx.mid.dt:
            raise ChanException("时序错误")

        if last.shape is Shape.G and fx.shape is Shape.D:
            bi = Bi(
                None,
                last,
                fx,
                cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
            )
            if bi.length > 4:
                eq = Bi.BI_EQUAL
                Bi.BI_EQUAL = False  # 起始点检测时不考虑相同起始点情况，避免递归
                if bi.real_high is not last.mid:
                    ##print("不是真顶")
                    top = bi.real_high
                    new = FenXing(
                        cklines[cklines.index(top) - 1],
                        top,
                        cklines[cklines.index(top) + 1],
                    )
                    assert new.shape is Shape.G, new
                    Bi.analyzer(
                        new, fxs, bis, cklines, _from, level=level + 1
                    )  # 处理新顶
                    Bi.analyzer(
                        fx, fxs, bis, cklines, _from, level=level + 1
                    )  # 再处理当前底
                    Bi.BI_EQUAL = eq
                    return
                Bi.BI_EQUAL = eq
                flag = bi.relation
                if flag and fx.mid is bi.real_low:
                    FenXing.append(fxs, fx)
                    Bi.append(bis, bi, _from)

                else:
                    ...
                    # 2024 05 21 修正
                    _cklines = cklines[last.mid.index :]
                    _fx, _bi = Bi.analysis_one(_cklines)

                    if _bi and len(fxs) > 2:
                        nb = Bi(
                            None,
                            fxs[-3],
                            _bi.start,
                            cklines[fxs[-3].mid.index : _bi.start.mid.index + 1],
                        )
                        if not nb.check():
                            return
                        # print(_bi)
                        tmp = fxs.pop()
                        assert tmp is last
                        Bi.pop(bis, tmp, _from)
                        Bi.analyzer(
                            _bi.start, fxs, bis, cklines, _from, level=level + 1
                        )

            else:
                ...
                # GD
                if right.high > last.speck:
                    tmp = fxs.pop()
                    assert tmp is last
                    Bi.pop(bis, tmp, _from)

        elif last.shape is Shape.D and fx.shape is Shape.G:
            bi = Bi(
                None,
                last,
                fx,
                cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
            )
            if bi.length > 4:
                eq = Bi.BI_EQUAL
                Bi.BI_EQUAL = False  # 起始点检测时不考虑相同起始点情况，避免递归
                if bi.real_low is not last.mid:
                    ##print("不是真底")
                    bottom = bi.real_low
                    new = FenXing(
                        cklines[cklines.index(bottom) - 1],
                        bottom,
                        cklines[cklines.index(bottom) + 1],
                    )
                    assert new.shape is Shape.D, new
                    Bi.analyzer(
                        new, fxs, bis, cklines, _from, level=level + 1
                    )  # 处理新底
                    Bi.analyzer(
                        fx, fxs, bis, cklines, _from, level=level + 1
                    )  # 再处理当前顶
                    Bi.BI_EQUAL = eq
                    return
                Bi.BI_EQUAL = eq
                flag = bi.relation
                if flag and fx.mid is bi.real_high:
                    FenXing.append(fxs, fx)
                    Bi.append(bis, bi, _from)

                else:
                    ...
                    # 2024 05 21 修正
                    _cklines = cklines[last.mid.index :]
                    _fx, _bi = Bi.analysis_one(_cklines)

                    if _bi and len(fxs) > 2:
                        nb = Bi(
                            None,
                            fxs[-3],
                            _bi.start,
                            cklines[fxs[-3].mid.index : _bi.start.mid.index + 1],
                        )
                        if not nb.check():
                            return
                        # print(_bi)
                        tmp = fxs.pop()
                        assert tmp is last
                        Bi.pop(bis, tmp, _from)
                        Bi.analyzer(
                            _bi.start, fxs, bis, cklines, _from, level=level + 1
                        )

            else:
                ...
                # DG
                if right.low < last.speck:
                    tmp = fxs.pop()
                    assert tmp is last
                    Bi.pop(bis, tmp, _from)

        elif last.shape is Shape.G and fx.shape is Shape.S:
            if last.speck < right.high:
                tmp = fxs.pop()
                Bi.pop(bis, tmp, _from)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.D
                    bottom = min(
                        cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                        key=lambda o: o.low,
                    )
                    assert bottom.shape is Shape.D
                    if last.speck > bottom.low:
                        tmp = fxs.pop()
                        Bi.pop(bis, tmp, _from)

                        new = FenXing(
                            cklines[cklines.index(bottom) - 1],
                            bottom,
                            cklines[cklines.index(bottom) + 1],
                        )
                        assert new.shape is Shape.D, new
                        Bi.analyzer(
                            new, fxs, bis, cklines, _from, level=level + 1
                        )  # 处理新底
                        ##print("GS修正")

        elif last.shape is Shape.D and fx.shape is Shape.X:
            if last.speck > right.low:
                tmp = fxs.pop()
                Bi.pop(bis, tmp, _from)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.G
                    top = max(
                        cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                        key=lambda o: o.high,
                    )
                    assert top.shape is Shape.G
                    if last.speck < top.high:
                        tmp = fxs.pop()
                        Bi.pop(bis, tmp, _from)

                        new = FenXing(
                            cklines[cklines.index(top) - 1],
                            top,
                            cklines[cklines.index(top) + 1],
                        )
                        assert new.shape is Shape.G, new
                        Bi.analyzer(
                            new, fxs, bis, cklines, _from, level=level + 1
                        )  # 处理新顶
                        ##print("DX修正")

        elif last.shape is Shape.G and fx.shape is Shape.G:
            if last.speck < fx.speck:
                tmp = fxs.pop()
                Bi.pop(bis, tmp, _from)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.D
                    bottom = min(
                        cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                        key=lambda o: o.low,
                    )
                    assert bottom.shape is Shape.D
                    if last.speck > bottom.low:
                        tmp = fxs.pop()
                        Bi.pop(bis, tmp, _from)

                        new = FenXing(
                            cklines[cklines.index(bottom) - 1],
                            bottom,
                            cklines[cklines.index(bottom) + 1],
                        )
                        assert new.shape is Shape.D, new
                        Bi.analyzer(
                            new, fxs, bis, cklines, _from, level=level + 1
                        )  # 处理新底
                        Bi.analyzer(
                            fx, fxs, bis, cklines, _from, level=level + 1
                        )  # 再处理当前顶
                        ##print("GG修正")
                        return

                if not fxs:
                    FenXing.append(fxs, fx)
                    return
                bi = Bi(
                    None,
                    fxs[-1],
                    fx,
                    cklines[cklines.index(fxs[-1].mid) : cklines.index(fx.mid) + 1],
                )
                FenXing.append(fxs, fx)
                Bi.append(bis, bi, _from)

        elif last.shape is Shape.D and fx.shape is Shape.D:
            if last.speck > fx.speck:
                tmp = fxs.pop()
                Bi.pop(bis, tmp, _from)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.G
                    top = max(
                        cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                        key=lambda o: o.high,
                    )
                    assert top.shape is Shape.G
                    if last.speck < top.high:
                        tmp = fxs.pop()
                        Bi.pop(bis, tmp, _from)

                        new = FenXing(
                            cklines[cklines.index(top) - 1],
                            top,
                            cklines[cklines.index(top) + 1],
                        )
                        assert new.shape is Shape.G, new
                        Bi.analyzer(
                            new, fxs, bis, cklines, _from, level=level + 1
                        )  # 处理新顶
                        Bi.analyzer(
                            fx, fxs, bis, cklines, _from, level=level + 1
                        )  # 再处理当前底
                        ##print("DD修正")
                        return

                if not fxs:
                    FenXing.append(fxs, fx)
                    return
                bi = Bi(
                    None,
                    fxs[-1],
                    fx,
                    cklines[cklines.index(fxs[-1].mid) : cklines.index(fx.mid) + 1],
                )
                FenXing.append(fxs, fx)
                Bi.append(bis, bi, _from)

        elif last.shape is Shape.G and fx.shape is Shape.X:
            ...
        elif last.shape is Shape.D and fx.shape is Shape.S:
            ...
        else:
            raise ChanException(last.shape, fx.shape)

    def analysis_one(cklines: List[NewBar]) -> tuple[Optional[FenXing], Optional["Bi"]]:
        try:
            cklines[2]
        except IndexError:
            return None, None
        bis = []
        fxs = []
        fx = None
        size = len(cklines)
        for i in range(1, size - 2):
            left, mid, right = cklines[i - 1], cklines[i], cklines[i + 1]
            fx = FenXing(left, mid, right)
            Bi.analyzer(fx, fxs, bis, cklines, "tmp")
            if bis:
                return fx, bis[0]
        if bis:
            return fx, bis[0]
        return None, None


class FeatureSequence(Observable, Observer):
    CAN = True
    CMD_APPEND = "feature_append"
    CMD_MODIFY = "feature_modify"
    CMD_REMOVE = "feature_remove"

    def __init__(self, elements: set, direction: Direction):
        super().__init__()
        self.__shape_id = TVShapeID()
        self.__elements: set = elements
        self.direction: Direction = direction  # 线段方向
        self.shape: Optional[Shape] = None
        self.index = 0
        self.attach_observer(self)

    @property
    def shape_id(self) -> str:
        return self.__shape_id.shape_id

    def copy(self, other: "FeatureSequence") -> "FeatureSequence":
        self.__elements = other.__elements
        self.direction = other.direction
        self.shape = other.shape
        self.index = other.index
        # self.notify_observer(cmd=FeatureSequence.CMD_MODIFY, obj=self)
        return self

    def update(self, observable: "Observable", **kwords: Any):
        if not FeatureSequence.CAN:
            return
        cmd = kwords.get("cmd")
        points = [
            {"time": 0, "price": 0},
            {"time": 0, "price": 0},
        ]
        if cmd != FeatureSequence.CMD_REMOVE:
            points = [
                {"time": int(self.start.dt.timestamp()), "price": self.start.speck},
                {"time": int(self.end.dt.timestamp()), "price": self.end.speck},
            ]
        options = {
            "shape": "trend_line",
            # "showInObjectsTree": True,
            # "disableSave": False,
            # "disableSelection": True,
            # "disableUndo": False,
            # "filled": True,
            # "lock": False,
            "text": "feature",
        }
        properties = {
            "linecolor": "#22FF22" if self.direction is Direction.Down else "#ff2222",
            "linewidth": 2,
            "linestyle": 2,
            # "showLabel": True,
            "title": "特征序列",
        }

        message = {
            "type": "shape",
            "cmd": cmd,
            "name": "trend_line",
            "id": self.shape_id,
            "points": points,
            "options": options,
            "properties": properties,
        }

        if cmd in (
            FeatureSequence.CMD_APPEND,
            FeatureSequence.CMD_REMOVE,
            FeatureSequence.CMD_MODIFY,
        ):
            # 后端实现 增 删 改
            future = asyncio.run_coroutine_threadsafe(
                Observer.queue.put(message), Observer.loop
            )
            try:
                future.result()  # 确保任务已添加到队列
            except Exception as e:
                print(f"Error adding task to queue: {e}")
        super().update(observable, **kwords)

    def __str__(self):
        if not self.__elements:
            return f"空特征序列({self.direction})"
        return f"特征序列({self.direction}, {self.start.dt}, {self.end.dt}, {len(self.__elements)})"

    def __repr__(self):
        if not self.__elements:
            return f"空特征序列({self.direction})"
        return f"特征序列({self.direction}, {self.start.dt}, {self.end.dt}, {len(self.__elements)})"

    def __len__(self):
        return len(self.__elements)

    def __iter__(self):
        return iter(self.__elements)

    def add(self, obj: Union[Bi, "Duan"], _from):
        direction = Direction.Down if self.direction is Direction.Up else Direction.Up
        if obj.direction is not direction:
            raise ChanException("方向不匹配", direction, obj, self)
        self.__elements.add(obj)
        if _from == "analyzer":
            self.notify_observer(cmd=FeatureSequence.CMD_MODIFY, obj=self)

    def remove(self, obj: Union[Bi, "Duan"], _from):
        direction = Direction.Down if self.direction is Direction.Up else Direction.Up
        if obj.direction is not direction:
            raise ChanException("方向不匹配", direction, obj, self)
        try:
            self.__elements.remove(obj)
        except Exception as e:
            # print(self)
            raise e
        if self.__elements:
            if _from == "analyzer":
                self.notify_observer(cmd=FeatureSequence.CMD_MODIFY, obj=self)
        else:
            if _from == "analyzer":
                self.notify_observer(cmd=FeatureSequence.CMD_REMOVE, obj=self)

    @property
    def start(self) -> FenXing:
        if not self.__elements:
            raise ChanException("数据异常", self)
        func = min
        if self.direction is Direction.Up:  # 线段方向向上特征序列取高高
            func = max
        if self.direction is Direction.Down:
            func = min
        fx = func([obj.start for obj in self.__elements], key=lambda o: o.speck)
        assert fx.shape in (Shape.G, Shape.D)
        return fx

    @property
    def end(self) -> FenXing:
        if not self.__elements:
            raise ChanException("数据异常", self)
        func = min
        if self.direction is Direction.Up:  # 线段方向向上特征序列取高高
            func = max
        if self.direction is Direction.Down:
            func = min
        fx = func([obj.end for obj in self.__elements], key=lambda o: o.speck)
        assert fx.shape in (Shape.G, Shape.D)
        return fx

    @property
    def high(self) -> float:
        return max([self.end, self.start], key=lambda fx: fx.speck).speck

    @property
    def low(self) -> float:
        return min([self.end, self.start], key=lambda fx: fx.speck).speck

    @staticmethod
    def analyzer(bis: list, direction: Direction, _from):
        result: List[FeatureSequence] = []
        for obj in bis:
            if obj.direction is direction:
                continue
            if result:
                last = result[-1]

                if double_relation(last, obj) in (Direction.Left,):
                    last.add(obj, _from)
                else:
                    result.append(
                        FeatureSequence(
                            {obj},
                            (
                                Direction.Up
                                if obj.direction is Direction.Down
                                else Direction.Down
                            ),
                        )
                    )
                    # dp("FS.ANALYSIS", double_relation(last, obj))
            else:
                result.append(
                    FeatureSequence(
                        {obj},
                        (
                            Direction.Up
                            if obj.direction is Direction.Down
                            else Direction.Down
                        ),
                    )
                )
        return result


class Duan(BaseChanObject, Observer):
    OBJS: List["Duan"] = []
    DUAN_OBJS: List["Duan"] = []  # 段的段
    FAKE = None
    CMD_APPEND = "duan_append"
    CMD_MODIFY = "duan_modify"
    CMD_REMOVE = "duan_remove"
    CDM_ZS_OBSERVER = "CDM_ZS_OBSERVER"

    # __slots__ =
    def __init__(
        self,
        pre: Optional["Duan"],
        start: FenXing,
        end: Union[FenXing, NewBar],
        elements: List[Bi],
    ) -> None:
        super().__init__()
        self._type = elements[-1].__class__.__name__
        self._features: list[Optional[FeatureSequence]] = [None, None, None]
        if start.shape is Shape.G:
            self.direction = Direction.Down
            self.high = start.speck
            self.low = end.low
        elif start.shape is Shape.D:
            self.direction = Direction.Up
            self.high = end.high
            self.low = start.speck
        else:
            raise ChanException(start.shape, end.shape)
        for i in range(1, len(self.elements)):
            assert self.elements[i - 1].index + 1 == self.elements[i].index, (
                self.elements[i - 1].index,
                self.elements[i].index,
            )
        if pre is not None:
            assert pre.end is start, (pre.end, start)
            self.index = pre.index + 1
        self.pre = pre
        self.__start = start
        self.__end = end
        self.elements = elements
        self.jump: bool = False  # 特征序列是否有缺口

        self.attach_observer(self)

    def update(self, observable: "Observable", **kwords: Any):
        # 实现 自我观察
        cmd = kwords.get("cmd")
        obj = kwords.get("obj")
        if cmd == f"Duan_{BaseChanObject.CMD_DONE}":
            Duan.DUAN_OBJS and Duan.DUAN_OBJS[-1].notify_observer(
                cmd=Duan.CMD_MODIFY, obj=Duan.DUAN_OBJS[-1]
            )

        points = [
            {"time": int(self.start.dt.timestamp()), "price": self.start.speck},
            {
                "time": int(self.end.dt.timestamp()),
                "price": self.end.speck,
            },
        ]
        options = {
            "shape": "trend_line",
            # "showInObjectsTree": True,
            # "disableSave": False,
            # "disableSelection": True,
            # "disableUndo": False,
            # "filled": True,
            # "lock": False,
            "text": "duan",
        }
        properties = {
            "linecolor": "#F1C40F" if self._type == "Bi" else "#00C40F",
            "linewidth": 3,
            "title": f"Duan-{self.index}",
            "text": "duan",
        }

        message = {
            "type": "shape",
            "cmd": cmd,
            "name": "trend_line",
            "id": self.shape_id,
            "points": points,
            "options": options,
            "properties": properties,
        }
        if cmd in (Duan.CMD_APPEND, Duan.CMD_REMOVE, Duan.CMD_MODIFY):
            # 后端实现 增 删 改
            if cmd == Duan.CMD_REMOVE:
                self.left = None
                self.right = None
                self.mid = None
            future = asyncio.run_coroutine_threadsafe(
                Observer.queue.put(message), Observer.loop
            )

            try:
                future.result()  # 确保任务已添加到队列
            except Exception as e:
                print(f"Error adding task to queue: {e}")
        super().update(observable, **kwords)

    def __str__(self):
        return f"Duan({self.index}, {self.direction}, {len(self.elements)}, 完成否:{self.done}, {self.pre is not None}, {self.start}, {self.end}, {self._type})"

    def __repr__(self):
        return f"Duan({self.index}, {self.direction}, {len(self.elements)}, 完成否:{self.done}, {self.pre is not None}, {self.start}, {self.end}, {self._type})"

    @property
    def lmr(self) -> tuple[bool, bool, bool]:
        return self.left is not None, self.mid is not None, self.right is not None

    @property
    def state(self) -> Optional[States]:
        if self.pre is not None:
            if self.pre.mid is None:
                return None
            relation = double_relation(self.pre.left, self.pre.mid)
            if relation is Direction.JumpUp and self.direction is Direction.Up:
                return "老阳"
            elif relation is Direction.JumpDown and self.direction is Direction.Down:
                return "老阴"
            else:
                return "小阳" if self.direction is Direction.Up else "少阴"
        else:
            return "小阳" if self.direction is Direction.Up else "少阴"

    def __feature_setter(self, offset: int, feature: Optional[FeatureSequence]):
        if feature is None:
            if self._features[offset]:
                self._features[offset].notify_observer(
                    cmd=FeatureSequence.CMD_REMOVE, obj=self._features[offset]
                )
            self._features[offset] = feature
            return

        if self._features[offset] is None:
            feature.notify_observer(cmd=FeatureSequence.CMD_APPEND, obj=feature)
            self._features[offset] = feature
        else:
            self._features[offset].copy(feature)
            self._features[offset].notify_observer(
                cmd=FeatureSequence.CMD_MODIFY, obj=self._features[offset]
            )

    @property
    def left(self) -> "FeatureSequence":
        return self._features[0]

    @left.setter
    def left(self, feature: FeatureSequence):
        self.__feature_setter(0, feature)

    @property
    def mid(self) -> "FeatureSequence":
        return self._features[1]

    @mid.setter
    def mid(self, feature: FeatureSequence):
        self.__feature_setter(1, feature)
        # if self.mid is not None:
        #    self.end = self.mid.start

    @property
    def right(self) -> "FeatureSequence":
        return self._features[2]

    @right.setter
    def right(self, feature: FeatureSequence):
        self.__feature_setter(2, feature)

    @property
    def start(self) -> FenXing:
        return self.__start

    @start.setter
    def start(self, start: FenXing):
        self.__start = start

    @property
    def end(self) -> FenXing:
        return self.__end

    @end.setter
    def end(self, end: Union[FenXing, NewBar]):
        # old = self.__end
        if self.__end is end:
            return
        self.__end = end
        if self.direction is Direction.Up:
            self.high = end.speck
        elif self.direction is Direction.Down:
            self.low = end.speck
        else:
            raise ChanException
        self.notify_observer(cmd=Duan.CMD_MODIFY, obj=self)

    def get_elements(self) -> Iterable[Bi]:
        elements = []
        for obj in self.elements:
            elements.append(obj)
            if obj.end is self.end:
                break
        return elements

    def append_element(self, bi: Bi):
        if self.elements[-1].end is bi.start:
            if bi.direction is not self.direction:
                if len(self.elements) >= 3:
                    if self.direction is Direction.Down:
                        if self.elements[-3].low < bi.high:
                            ddp("向下线段被笔破坏", bi)
                    if self.direction is Direction.Up:
                        if self.elements[-3].high > bi.low:
                            ddp("向上线段被笔破坏", bi)
            self.elements.append(bi)

        else:
            dp("线段添加元素时，元素不连续", self.elements[-1], bi)
            raise ChanException("线段添加元素时，元素不连续", self.elements[-1], bi)

    def pop_element(self, bi: Bi) -> bool:
        if self.elements[-1] is bi:
            self.elements.pop()
            return True
        else:
            raise ChanException("线段弹出元素时，元素不匹配")

    def get_features(
        self,
    ) -> Tuple[
        Optional[FeatureSequence], Optional[FeatureSequence], Optional[FeatureSequence]
    ]:
        features = FeatureSequence.analyzer(self.elements, self.direction, "tmp")
        if len(features) == 0:
            return None, None, None
        if len(features) == 1:
            return features[0], None, None
        if len(features) == 2:
            return features[0], features[1], None
        if len(features) >= 3:
            if self.direction is Direction.Up:
                if double_relation(features[-2], features[-1]) in (
                    Direction.Down,
                    Direction.JumpDown,
                ):
                    return features[-3], features[-2], features[-1]
                else:
                    return features[-2], features[-1], None
            else:
                if double_relation(features[-2], features[-1]) in (
                    Direction.Up,
                    Direction.JumpUp,
                ):
                    return features[-3], features[-2], features[-1]
                else:
                    return features[-2], features[-1], None

    def set_done(self, fx: FenXing):
        self.left, self.mid, self.right = self.get_features()
        assert fx is self.mid.start

        elements = []
        for obj in self.elements:
            if elements:
                elements.append(obj)
            if obj.start is fx:
                elements.append(obj)
        self.end = fx
        self.done = True  # 注意观察者！！
        self.jump = double_relation(self.left, self.mid) in (
            Direction.JumpUp,
            Direction.JumpDown,
        )
        return elements

    @staticmethod
    def append(xds, duan, _from):
        if xds and xds[-1].end is not duan.start:
            raise ChanException("线段连续性错误")
        i = 0
        pre = None
        if xds:
            i = xds[-1].index + 1
            pre = xds[-1]
        duan.index = i
        duan.pre = pre

        xds.append(duan)
        if _from == "analyzer":
            duan.notify_observer(cmd=Duan.CMD_APPEND, obj=duan)

        zss = ZhongShu.DUAN_OBJS
        if duan._type == "Duan":
            zss = ZhongShu.DUAN_DUAN_OBJS
        ZhongShu.analyzer_push(duan, zss, xds, 0, _from)

        # if duan._type == "Bi":
        #    Duan.analyzer_append(duan, Duan.DUAN_OBJS, 0, _from)

    @staticmethod
    def pop(xds, duan, _from):
        if not xds:
            return
        if not (xds[-1] is duan):
            raise ChanException

        duan = xds.pop()
        if _from == "analyzer":
            duan.notify_observer(cmd=Duan.CMD_REMOVE, obj=duan)

        zss = ZhongShu.DUAN_OBJS
        if duan._type == "Duan":
            zss = ZhongShu.DUAN_DUAN_OBJS
        ZhongShu.analyzer_pop(duan, zss, 0, _from)

        # if duan._type == "Bi":
        #    Duan.analyzer_pop(duan, Duan.DUAN_OBJS, 0, _from)
        return duan

    @staticmethod
    def analyzer_pop(bi, xds: List["Duan"], level, _from):
        ddp()
        cmd = "Duans.POP"
        duan: Duan = xds[-1]
        state: Optional[States] = duan.state
        last: Optional[Duan] = duan.pre
        last = xds[-2] if len(xds) > 1 else last

        lmr: Tuple[bool, bool, bool] = duan.lmr

        ddp("    " * level, cmd, state, lmr, duan, bi)
        # ddp("    " * level, duan.features)
        # ddp("    " * level, duan.elements)

        duan.pop_element(bi)

        if last is not None:
            if (last.right and bi in last.right) or (
                last.right is None and bi in last.left
            ):
                Duan.pop(xds, duan, _from)
                last.pop_element(bi)
                # last.features = [last.left, last.mid, None]
                last.right = None
                return

        if duan.elements:
            duan.left, duan.mid, duan.right = duan.get_features()
        else:
            Duan.pop(xds, duan, _from)
        return

    @staticmethod
    def analyzer_append(bi: Bi, xds: list["Duan"], level: int, _from: str):
        cmd = "Duans.PUSH"
        new = Duan(None, bi.start, bi.end, [bi])
        if not xds:
            Duan.append(xds, new, _from)
            return
        duan: Duan = xds[-1]
        state = duan.state
        # last: Optional[Duan] = duan.pre
        # last = duans[-2] if len(duans) > 1 else last
        # left: Optional[FeatureSequence] = duan.left
        # mid: Optional[FeatureSequence] = duan.mid
        # right: Optional[FeatureSequence] = duan.features[2]
        lmr: Tuple[bool, bool, bool] = duan.lmr

        ddp("    " * level, cmd, state, lmr, duan, bi)
        ddp("    " * level, duan._features)
        ddp("    " * level, duan.elements)

        if duan.direction is bi.direction:
            duan.append_element(bi)
            ddp("    " * level, "方向相同, 更新结束点", duan.end, duan.state)
            if duan.mid:
                if duan.direction is Direction.Up:
                    if duan.high < bi.high:
                        duan.end = bi.end
                    else:
                        duan.end = duan.mid.start
                else:
                    if duan.low > bi.low:
                        duan.end = bi.end
                    else:
                        duan.end = duan.mid.start
            else:
                duan.end = bi.end
            return

        if len(xds) == 1:
            if (
                (duan.direction is Direction.Up)
                and (bi.low < duan.start.speck)
                and len(duan.elements) == 1
            ):
                new = new = Duan(None, bi.start, bi.end, [bi])
                Duan.pop(xds, duan, _from)
                Duan.append(xds, new, _from)
                return
            if (
                (duan.direction is Direction.Down)
                and (bi.high > duan.start.speck)
                and len(duan.elements) == 1
            ):
                new = new = Duan(None, bi.start, bi.end, [bi])
                Duan.pop(xds, duan, _from)
                Duan.append(xds, new, _from)
                return

        duan.append_element(bi)
        l, m, r = duan.get_features()
        if r:
            # Duan.pop(xds, duan, _from)
            # Duan.append(xds, duan, _from)
            elements = duan.set_done(m.start)
            new = Duan(duan, elements[0].start, elements[-1].end, elements)
            Duan.append(xds, new, _from)
            new.left, new.mid, new.right = new.get_features()
            if duan.direction is Direction.Up:
                fx = "顶分型"
            else:
                fx = "底分型"

            ddp("    " * level, f"{fx}终结, 缺口: {duan.jump}")

        else:
            duan.left, duan.mid, duan.right = duan.get_features()


class ZhongShu(BaseChanObject, Observer):
    OBJS: List["ZhongShu"] = []
    BI_OBJS: List["ZhongShu"] = []
    DUAN_OBJS: List["ZhongShu"] = []
    DUAN_DUAN_OBJS: List["ZhongShu"] = []
    TYPE: Tuple["str"] = ("Bi", "Duan", "ZouShi")

    CMD_APPEND = "zs_append"
    CMD_MODIFY = "zs_modify"
    CMD_REMOVE = "zs_remove"

    # __slots__ = "elements", "index", "level"

    def __init__(
        self, _type: str, direction: Direction, obj: Union[Bi, Duan, RawBar, NewBar]
    ):
        super().__init__()
        self.__direction = direction
        self.elements = [obj]
        self.level = 0
        self._type = _type
        self.__third: Optional[Union[Bi, Duan]] = None
        if type(obj) is Bi:
            self.level = 1
        if type(obj) is Duan:
            self.level = 2
        self.fake = None
        self.attach_observer(self)

    def copy_zs(self, zs: "ZhongShu"):
        self.elements = zs.elements
        self.fake = zs.fake
        self.level = zs.level
        self.third = zs.third
        self._type = zs._type
        self.done = zs.done
        self.__direction = zs.direction

    def __str__(self):
        return f"中枢({self.index}, {self.direction}, {self.zg}, {self.zd}, elements size={len(self.elements)}, {self.last_element})"

    def __repr__(self):
        return f"中枢({self.index}, {self.direction}, {self.zg}, {self.zd}, elements size={len(self.elements)}, {self.last_element})"

    def __eq__(self, other: "ZhongShu") -> bool:
        # `__eq__` is an instance method, which also accepts
        # one other object as an argument.

        if (
            type(other) == type(self)
            and other.elements == self.elements
            and other.direction == self.direction
            and other.third == self.third
        ):
            return True
        else:
            return False  # 返回False这一步也是需要写的哈，不然判断失败就没有返回值了

    def update(self, observable: "Observable", **kwords: Any):
        cmd = kwords.get("cmd")
        obj: Observable = kwords.get("obj")
        if cmd == f"{self.__class__.__name__}_{BaseChanObject.CMD_DONE}":
            self.notify_observer(cmd=ZhongShu.CMD_MODIFY, obj=self)
            return

        if cmd == f"{Duan.__name__}_{BaseChanObject.CMD_DONE}":
            self.notify_observer(cmd=ZhongShu.CMD_MODIFY, obj=self)
            obj.detach_observer(self)
            return

        if cmd == Duan.CDM_ZS_OBSERVER:
            self.notify_observer(cmd=ZhongShu.CMD_MODIFY, obj=self)
            return

        if cmd == Duan.CMD_MODIFY:  # 线段end改变事件
            if obj is self.third:
                if double_relation(self, self.third) in (
                    Direction.JumpUp,
                    Direction.JumpDown,
                ):
                    self.third = obj
                    return
                self.elements.append(self.third)
                self.third = None
            else:
                assert self.elements[-1] is obj
                self.notify_observer(cmd=ZhongShu.CMD_MODIFY, obj=self)
            return
        points = []
        if cmd == ZhongShu.CMD_REMOVE:
            points = []
        if cmd == ZhongShu.CMD_APPEND:
            points = (
                [
                    {"time": int(self.start.dt.timestamp()), "price": self.zg},
                    {
                        "time": int(self.elements[-1].end.mid.dt.timestamp()),
                        "price": self.zd,
                    },
                ]
                if len(self.elements) <= 3
                else [
                    {"time": int(self.start.dt.timestamp()), "price": self.zg},
                    {
                        "time": int(self.elements[-1].start.mid.dt.timestamp()),
                        "price": self.zd,
                    },
                ]
            )
        if cmd == ZhongShu.CMD_MODIFY:
            if self.left is None:
                return
            points = (
                [
                    {"time": int(self.start.dt.timestamp()), "price": self.zg},
                    {
                        "time": int(self.elements[-1].end.mid.dt.timestamp()),
                        "price": self.zd,
                    },
                ]
                if len(self.elements) <= 3
                else [
                    {"time": int(self.start.dt.timestamp()), "price": self.zg},
                    {
                        "time": int(self.elements[-1].start.mid.dt.timestamp()),
                        "price": self.zd,
                    },
                ]
            )

        options = {
            "shape": "rectangle",
            "text": f"zs",
        }
        properties = {
            "color": (
                "#993333" if self.direction is Direction.Down else "#99CC99"
            ),  # 上下上 为 红色，反之为 绿色
            "fillBackground": True,
            "backgroundColor": (
                "rgba(156, 39, 176, 0.2)"
                if self.level == 1
                else "rgba(1, 39, 176, 0.2)"
            ),
            "linewidth": 1 if self.level == 1 else 2,
            "transparency": 50,
            "showLabel": False,
            "horzLabelsAlign": "left",
            "vertLabelsAlign": "bottom",
            "textColor": "#9c27b0",
            "fontSize": 14,
            "bold": False,
            "italic": False,
            "extendLeft": False,
            "extendRight": not self.done,
            "visible": True,
            "frozen": False,
            "intervalsVisibilities": {
                "ticks": True,
                "seconds": True,
                "secondsFrom": 1,
                "secondsTo": 59,
                "minutes": True,
                "minutesFrom": 1,
                "minutesTo": 59,
                "hours": True,
                "hoursFrom": 1,
                "hoursTo": 24,
                "days": True,
                "daysFrom": 1,
                "daysTo": 366,
                "weeks": True,
                "weeksFrom": 1,
                "weeksTo": 52,
                "months": True,
                "monthsFrom": 1,
                "monthsTo": 12,
                "ranges": True,
            },
            "title": f"{self._type}中枢-{self.index}",
            "text": f"{self._type}zs",
        }

        message = {
            "type": "shape",
            "cmd": cmd,
            "name": "rectangle",
            "id": self.shape_id,
            "points": points,
            "options": options,
            "properties": properties,
        }

        if cmd in (ZhongShu.CMD_APPEND, ZhongShu.CMD_REMOVE, ZhongShu.CMD_MODIFY):
            # 后端实现 增 删 改
            ##print(message)
            future = asyncio.run_coroutine_threadsafe(
                Observer.queue.put(message), Observer.loop
            )

            try:
                future.result()  # 确保任务已添加到队列
            except Exception as e:
                print(f"Error adding task to queue: {e}")

    @property
    def third(self):
        return self.__third

    @third.setter
    def third(self, third):
        old = self.__third
        self.__third = third
        if third is not None:
            self.done = True
            if third.end.shape is Shape.G:
                Observer.plot_bsp(self._type, third.end.mid, BSPoint.TS)
            else:
                Observer.plot_bsp(self._type, third.end.mid, BSPoint.TB)
        else:
            self.done = False

        if old:
            if old.end.shape is Shape.G:
                Observer.plot_bsp(self._type, old.end.mid, BSPoint.TS, False)
            else:
                Observer.plot_bsp(self._type, old.end.mid, BSPoint.TB, False)

    @property
    def left(self) -> Union[Bi, Duan, RawBar, NewBar]:
        return self.elements[0] if self.elements else None

    @property
    def mid(self) -> Union[Bi, Duan, RawBar, NewBar]:
        return self.elements[1] if len(self.elements) > 1 else None

    @property
    def right(self) -> Union[Bi, Duan, RawBar, NewBar]:
        return self.elements[2] if len(self.elements) > 2 else None

    @property
    def last_element(self) -> Union[Bi, Duan, RawBar, NewBar]:
        return self.elements[-1] if self.elements else None

    @property
    def direction(self) -> Direction:
        return self.__direction
        # return Direction.Down if self.start.shape is Shape.D else Direction.Up

    @property
    def zg(self) -> float:
        return min(self.elements[:3], key=lambda o: o.high).high

    @property
    def zd(self) -> float:
        return max(self.elements[:3], key=lambda o: o.low).low

    @property
    def g(self) -> float:
        return min(self.elements, key=lambda o: o.high).high

    @property
    def d(self) -> float:
        return max(self.elements, key=lambda o: o.low).low

    @property
    def gg(self) -> float:
        return max(self.elements, key=lambda o: o.high).high

    @property
    def dd(self) -> float:
        return min(self.elements, key=lambda o: o.low).low

    @property
    def high(self) -> float:
        return self.zg

    @property
    def low(self) -> float:
        return self.zd

    @property
    def start(self) -> FenXing:
        return self.left.start

    @property
    def end(self) -> FenXing:
        return self.elements[-1].end

    def check(self) -> bool:
        return double_relation(self.left, self.right) in (
            Direction.Down,
            Direction.Up,
            Direction.Left,
            Direction.Right,
        )

    def pop_element(self, obj: Union[Bi, Duan], _from):
        if self.last_element.start is obj.start:
            if self.last_element is not obj:
                dp("警告：中枢元素不匹配!!!", self.last_element, obj)
            self.elements.pop()
            if obj.end.mid.bsp:
                # print("消除", obj.end.mid)
                Observer.plot_bsp(self._type, obj.end.mid, obj.end.mid.bsp[-1], False)

            # if _from == "analyzer" and self._type == "Duan":
            #   self.notify_observer(cmd=ZhongShu.CMD_MODIFY)
        else:
            raise ChanException("中枢无法删除元素", self.last_element, obj)

    def append_element(self, obj: Union[Bi, Duan], _from):
        relation = double_relation(self, obj)
        if self.last_element.end is obj.start:
            size = len(self.elements)

            if size >= 3 and (size + 1) % 2 == 0:
                enter = (
                    Bi.OBJS[self.elements[0].index - 1]
                    if self._type == "Bi"
                    else Duan.OBJS[self.elements[0].index - 1]
                )
                if (
                    self.elements[1].direction is Direction.Up
                    and obj.direction is Direction.Up
                ):
                    # 卖
                    ##print("计算买卖点， 新高", enter.macd, obj.macd, enter, obj)
                    if self.gg < obj.high:
                        if calc_bc([enter], [obj]):
                            # 背驰
                            obj.end.mid.bsp.append(BSPoint.FS)
                            Observer.plot_bsp(self._type, obj.end.mid, BSPoint.FS)

                elif (
                    self.elements[1].direction is Direction.Down
                    and obj.direction is Direction.Down
                ):
                    ##print("计算买卖点， 新低", enter.macd, obj.macd, enter, obj)
                    # 买
                    if self.dd > obj.low:
                        if calc_bc([enter], [obj]):
                            # 背驰
                            obj.end.mid.bsp.append(BSPoint.FB)
                            Observer.plot_bsp(self._type, obj.end.mid, BSPoint.FB)

            self.elements.append(obj)
            # if _from == "analyzer" and self._type == "Duan":
            #    self.notify_observer(cmd=ZhongShu.CMD_MODIFY)
        else:
            raise ChanException("中枢无法添加元素", relation, self.last_element, obj)

    def charts(self):
        return [
            (
                [
                    self.start.mid.dt,
                    self.start.mid.dt,
                    self.elements[-1].start.mid.dt,
                    self.elements[-1].start.mid.dt,
                    self.start.mid.dt,
                ]
                if len(self.elements) > 3
                else [
                    self.start.mid.dt,
                    self.start.mid.dt,
                    self.end.mid.dt,
                    self.end.mid.dt,
                    self.start.mid.dt,
                ]
            ),
            [self.zg, self.zd, self.zd, self.zg, self.zg],
            (
                "#993333" if self.direction is Direction.Up else "#99CC99"
            ),  # 上下上 为 红色，反之为 绿色
            self.level,
        ]

    @staticmethod
    def append(zss: List["ZhongShu"], zs: "ZhongShu", _from):
        i = 0
        if zss:
            i = zss[-1].index + 1
        zs.index = i
        zss.append(zs)
        if _from == "analyzer":
            zs.notify_observer(cmd=ZhongShu.CMD_APPEND, obj=zs)

    @staticmethod
    def pop(zss: List["ZhongShu"], zs: "ZhongShu", _from) -> Optional["ZhongShu"]:
        if not zss:
            return
        if zss[-1] is zs:
            if _from == "analyzer":
                zs.notify_observer(cmd=ZhongShu.CMD_REMOVE, obj=zs)
            return zss.pop()

    @staticmethod
    def analyzer_pop(obj: Union[Bi, Duan], zss: List["ZhongShu"], level: int, _from):
        cmd = "ZS.POP"
        if not zss:
            return
        lzs = zss[-1]
        if obj in lzs.elements:
            lzs.pop_element(obj, _from)
            if len(lzs.elements) < 3:
                ZhongShu.pop(zss, lzs, _from)
        elif obj is lzs.third:
            obj.clear_observer()
            lzs.third = None

    @staticmethod
    def analyzer_push(
        obj: Union[Bi, Duan],  # 当前最新
        zss: List["ZhongShu"],
        objs: Union[List[Bi], List[Duan]],  #
        level: int,
        _from="analyzer",
    ):
        cmd = "ZS.PUSH"
        if not zss:
            if len(objs) >= 3:
                if double_relation(objs[-3], objs[-1]) in (
                    Direction.JumpUp,
                    Direction.JumpDown,
                ):
                    return
                new = ZhongShu(
                    type(obj).__name__,
                    Direction.Up if obj.direction is Direction.Down else Direction.Down,
                    obj,
                )
                new.elements = objs[-3:]
                if new._type == "Duan":
                    new.elements[-1].attach_observer(new)
                ZhongShu.append(zss, new, _from)
            return

        lzs = zss[-1]
        if double_relation(lzs, obj) in (Direction.JumpUp, Direction.JumpDown):
            if lzs.third is None:
                lzs.third = obj
                if lzs._type == "Duan":
                    obj.attach_observer(lzs)
            else:
                assert lzs.third is not obj
                if lzs.third.direction is obj.direction:
                    if double_relation(objs[-3], objs[-1]) in (
                        Direction.JumpUp,
                        Direction.JumpDown,
                    ):
                        return
                    new = ZhongShu(
                        type(obj).__name__,
                        (
                            Direction.Up
                            if obj.direction is Direction.Down
                            else Direction.Down
                        ),
                        obj,
                    )
                    new.elements = objs[-3:]
                    if new._type == "Duan":
                        new.elements[-1].attach_observer(new)
                    ZhongShu.append(zss, new, _from)

        else:
            if lzs.third is None:
                lzs.append_element(obj, _from)
            else:
                assert lzs.third is not obj
                if lzs.third.direction is obj.direction:
                    if double_relation(objs[-3], objs[-1]) in (
                        Direction.JumpUp,
                        Direction.JumpDown,
                    ):
                        return
                    new = ZhongShu(
                        type(obj).__name__,
                        (
                            Direction.Up
                            if obj.direction is Direction.Down
                            else Direction.Down
                        ),
                        obj,
                    )
                    new.elements = objs[-3:]
                    if new._type == "Duan":
                        new.elements[-1].attach_observer(new)
                    ZhongShu.append(zss, new, _from)

    @staticmethod
    def analyzer(elements: List[Union[Bi, Duan]]) -> tuple[bool, list]:
        if len(elements) < 3:
            return False, []
        zss: List[Union[Bi, Duan, ZhongShu]] = []
        for obj in elements:
            ZhongShu.analyzer_push(obj, zss, elements, 0, "tmp")
        return len(zss) > 0, zss


class ZouShi(BaseChanObject, Observer):
    OBJS: List["ZouShi"] = []
    CMD_APPEND = "append"
    CMD_MODIFY = "modify"
    CMD_REMOVE = "remove"

    def __init__(self, obj: Union[Bi, Duan, Self]):
        super().__init__()
        self.elements = [obj]
        self.zss: List[ZhongShu] = []

        self.attach_observer(self)

    @property
    def last_element(self) -> Union[Bi, Duan, RawBar, NewBar, ZhongShu]:
        return self.elements[-1] if self.elements else None

    def pop_element(self, obj: Union[Bi, Duan], _from):
        if self.last_element.start is obj.start:
            if self.last_element is not obj:
                dp("警告：走势元素不匹配!!!", self.last_element, obj)
            self.elements.pop()
        else:
            raise ChanException("走势无法删除元素", self.last_element, obj)

    def append_element(self, obj: Union[Bi, Duan], _from):
        relation = double_relation(self, obj)
        if self.last_element.end is obj.start:
            self.elements.append(obj)
        else:
            raise ChanException("走势无法添加元素", relation, self.last_element, obj)

    def update(self, observable: "Observable", **kwords: Any):
        cmd = kwords.get("cmd")

        super().update(observable, **kwords)


class KlineGenerator:
    def __init__(self, arr=[3, 2, 5, 3, 7, 4, 7, 2.5, 5, 4, 8, 6]):
        self.dt = datetime.datetime(2021, 9, 3, 19, 50, 40, 916152)
        self.arr = arr

    def up(self, start, end, size=8):
        n = 0
        m = round(abs(start - end) * (1 / size), 8)
        o = start
        # c = round(o + m, 4)

        while n < size:
            c = round(o + m, 4)
            yield RawBar(self.dt, o, c, o, c, 1)
            o = c
            n += 1
            self.dt = datetime.datetime.fromtimestamp(self.dt.timestamp() + 60 * 60)

    def down(self, start, end, size=8):
        n = 0
        m = round(abs(start - end) * (1 / size), 8)
        o = start
        # c = round(o - m, 4)

        while n < size:
            c = round(o - m, 4)
            yield RawBar(self.dt, o, o, c, c, 1)
            o = c
            n += 1
            self.dt = datetime.datetime.fromtimestamp(self.dt.timestamp() + 60 * 60)

    @property
    def result(self):
        size = len(self.arr)
        i = 0
        # sizes = [5 for i in range(l)]
        result = []
        while i + 1 < size:
            s = self.arr[i]
            e = self.arr[i + 1]
            if s > e:
                for k in self.down(s, e):
                    result.append(k)
            else:
                for k in self.up(s, e):
                    result.append(k)
            i += 1
        return result


def gen(arr) -> "CZSCAnalyzer":
    g = KlineGenerator(arr)

    c = CZSCAnalyzer("test", 60, [])
    for b in g.result:
        c.push(b)
    return c


class BaseAnalyzer:
    def __init__(self, symbol: str, freq: int):
        self.__symbol = symbol
        self.__freq = freq
        Observer.sigals.clear()
        Observer.sigal = None
        RawBar.OBJS = []
        NewBar.OBJS = []
        FenXing.OBJS = []
        Bi.OBJS = []
        Bi.FAKE = None
        Duan.OBJS = []
        Duan.DUAN_OBJS = []
        ZhongShu.OBJS = []
        ZhongShu.BI_OBJS = []
        ZhongShu.DUAN_OBJS = []
        ZhongShu.DUAN_DUAN_OBJS = []
        self._raws: List[RawBar] = RawBar.OBJS  # 原始K线列表
        self._news: List[NewBar] = NewBar.OBJS  # 去除包含关系K线列表
        self._fxs: List[FenXing] = FenXing.OBJS  # 分型列表
        self._bis: List[Bi] = Bi.OBJS  # 笔
        self._duans: List[Duan] = Duan.OBJS

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def freq(self) -> int:
        return self.__freq

    def push(
        self,
        bar: RawBar,
        fast_period: int = BaseChanObject.FAST,
        slow_period: int = BaseChanObject.SLOW,
        signal_period: int = BaseChanObject.SIGNAL,
    ):
        Observer.sigal = None
        last = self._news[-1] if self._news else None
        news = self._news
        if last is None:
            bar.to_new_bar(None)
        else:
            last.merge(bar)

        klines = RawBar.OBJS
        if len(klines) == 1:
            ema_slow = klines[-1].close
            ema_fast = klines[-1].close
        else:
            ema_slow = (
                2 * klines[-1].close
                + klines[-2].cache[f"ema_{slow_period}"] * (slow_period - 1)
            ) / (slow_period + 1)
            ema_fast = (
                2 * klines[-1].close
                + klines[-2].cache[f"ema_{fast_period}"] * (fast_period - 1)
            ) / (fast_period + 1)

        klines[-1].cache[f"ema_{slow_period}"] = ema_slow
        klines[-1].cache[f"ema_{fast_period}"] = ema_fast
        DIF = ema_fast - ema_slow
        klines[-1].cache[f"dif_{fast_period}_{slow_period}_{signal_period}"] = DIF

        if len(klines) == 1:
            dea = DIF
        else:
            dea = (
                2 * DIF
                + klines[-2].cache[f"dea_{fast_period}_{slow_period}_{signal_period}"]
                * (signal_period - 1)
            ) / (signal_period + 1)

        klines[-1].cache[f"dea_{fast_period}_{slow_period}_{signal_period}"] = dea
        macd = (DIF - dea) * 2
        klines[-1].cache[f"macd_{fast_period}_{slow_period}_{signal_period}"] = macd

        fx: Optional[FenXing] = NewBar.get_last_fx()
        if fx is not None:
            Bi.analyzer(fx, FenXing.OBJS, Bi.OBJS, NewBar.OBJS)
        Bi.calc_fake()

    def toCharts(self, path: str = "czsc.html", useReal=False):
        import echarts_plot  # type: ignore # czsc

        reload(echarts_plot)
        kline_pro = echarts_plot.kline_pro
        fx = [
            {"dt": fx.dt, "fx": fx.low if fx.shape is Shape.D else fx.high}
            for fx in self._fxs
        ]
        bi = [
            {"dt": fx.dt, "bi": fx.low if fx.shape is Shape.D else fx.high}
            for fx in self._fxs
        ]

        # xd = [{"dt": fx.dt, "xd": fx.low if fx.shape is Shape.D else fx.high} for fx in self.xd_fxs]

        xd = []
        mergers = []
        for duan in self._duans:
            xd.extend(
                [
                    {"xd": duan.start.speck, "dt": duan.start.dt},
                    {"xd": duan.end.speck, "dt": duan.end.dt},
                ]
            )
            left, mid, right = duan._features
            if left:
                if len(left) > 1:
                    mergers.append(left)
            if mid:
                if len(mid) > 1:
                    mergers.append(mid)
            if right:
                if len(right) > 1:
                    mergers.append(right)
            else:
                print("right is None")

        dzs = [zs.charts() for zs in ZhongShu.DUAN_OBJS if len(zs.elements) >= 3]
        bzs = [zs.charts() for zs in ZhongShu.BI_OBJS if len(zs.elements) >= 3]

        charts = kline_pro(
            (
                [
                    {
                        "dt": x.dt,
                        "open": x.open,
                        "high": x.high,
                        "low": x.low,
                        "close": x.close,
                        "vol": x.volume,
                    }
                    for x in self._raws
                ]
                if useReal
                else [
                    {
                        "dt": x.dt,
                        "open": x.open,
                        "high": x.high,
                        "low": x.low,
                        "close": x.close,
                        "vol": x.volume,
                    }
                    for x in self._news
                ]
            ),
            fx=fx,
            bi=bi,
            xd=xd,
            mergers=mergers,
            bzs=bzs,
            dzs=dzs,
            title=self.symbol + "-" + str(self.freq / 60) + "分钟",
            width="100%",
            height="80%",
        )

        charts.render(path)
        return charts


class CZSCAnalyzer:
    def __init__(self, symbol: str, freq: int, freqs: List[int] = None):
        if freqs is None:
            freqs = [freq]
        else:
            freqs.append(freq)
            freqs = list(set(freqs))
        self.symbol = symbol
        self.freq = freq
        self.freqs = freqs

        self._analyzeies = dict()
        self.__analyzer = BaseAnalyzer(symbol, freq)
        self.raws = RawBar.OBJS

    def toCharts(self, path: str = "czsc.html", useReal=False):
        return self.__analyzer.toCharts(path=path, useReal=useReal)

    @property
    def news(self):
        return self.__analyzer._news

    @final
    def step(
        self,
        dt: datetime.datetime | int | str,
        open: float | str,
        high: float | str,
        low: float | str,
        close: float | str,
        volume: float | str,
    ):
        if type(dt) is datetime.datetime:
            ...
        elif isinstance(dt, str):
            dt: datetime.datetime = datetime.datetime.fromtimestamp(int(dt))
        elif isinstance(dt, int):
            dt: datetime.datetime = datetime.datetime.fromtimestamp(dt)
        else:
            raise ChanException("类型不支持", type(dt))
        open = float(open)
        high = float(high)
        low = float(low)
        close = float(close)
        volume = float(volume)

        index = 0

        last = RawBar(
            dt=dt,
            o=open,
            h=high,
            l=low,
            c=close,
            v=volume,
            i=index,
        )
        self.push(last)

    def push(self, k: RawBar):
        if Observer.CAN:
            time.sleep(Observer.TIME)
        try:
            self.__analyzer.push(k)
        except Exception as e:
            # self.__analyzer.toCharts()
            # with open(f"{self.symbol}-{int(self._bars[0].dt.timestamp())}-{int(self._bars[-1].dt.timestamp())}.dat", "wb") as f:
            #    f.write(self.save_bytes())
            raise e

    @classmethod
    def load_bytes(cls, symbol: str, bytes_data: bytes, freq: int) -> "Self":
        size = struct.calcsize(">6d")
        obj = cls(symbol, freq)
        bytes_data = bytes_data  #    [size * 600 : size * 743]
        while bytes_data:
            t = bytes_data[:size]
            k = RawBar.from_bytes(t)
            obj.push(k)
            bytes_data = bytes_data[size:]
            if len(bytes_data) < size:
                break
            if Observer.CAN and Observer.thread is None:
                break
        return obj

    def save_bytes(self) -> bytes:
        data = b""
        for k in self.raws:
            data += bytes(k)
        return data

    def save_file(self):
        with open(
            f"{self.symbol}-{self.freq}-{int(self.__analyzer._raws[0].dt.timestamp())}-{int(self.__analyzer._raws[-1].dt.timestamp())}.dat",
            "wb",
        ) as f:
            f.write(self.save_bytes())

    @classmethod
    def load_file(cls, path: str) -> "Self":
        name = Path(path).name.split(".")[0]
        symbol, freq, s, e = name.split("-")
        with open(path, "rb") as f:
            dat = f.read()
            return cls.load_bytes(symbol, dat, int(freq))


class Bitstamp(CZSCAnalyzer):
    """ """

    def __init__(self, symbol: str, freq: Union[Freq, int, str], size: int = 0):
        if type(freq) is Freq:
            super().__init__(symbol, freq.value)
            self.freq: int = freq.value
        elif type(freq) is int:
            super().__init__(symbol, freq)
            self.freq: int = freq
        elif type(freq) is str:
            super().__init__(symbol, int(freq))
            self.freq: int = int(freq)
        else:
            raise ChanException

    def init(self, size):
        self.left_date_timestamp: int = int(datetime.datetime.now().timestamp() * 1000)
        left = int(self.left_date_timestamp / 1000) - self.freq * size
        if left < 0:
            raise ChanException
        _next = left
        while 1:
            data = self.ohlc(
                self.symbol, self.freq, _next, _next := _next + self.freq * 1000
            )
            if not data.get("data"):
                # print(data)
                raise ChanException
            for bar in data["data"]["ohlc"]:
                try:
                    self.step(
                        bar["timestamp"],
                        bar["open"],
                        bar["high"],
                        bar["low"],
                        bar["close"],
                        bar["volume"],
                    )
                except ChanException as e:
                    # continue
                    self.save_file()
                    raise e

            # start = int(data["data"]["ohlc"][0]["timestamp"])
            end = int(data["data"]["ohlc"][-1]["timestamp"])

            _next = end
            if len(data["data"]["ohlc"]) < 100:
                break
            if Observer.CAN and Observer.thread is None:
                break
        Observer.thread = None

    @staticmethod
    def ohlc(pair: str, step: int, start: int, end: int, length: int = 1000) -> Dict:
        proxies = {
            "http": "http://127.0.0.1:11809",
            "https": "http://127.0.0.1:11809",
        }
        s = requests.Session()

        s.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
            "content-type": "application/json",
        }
        url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/?step={step}&limit={length}&start={start}&end={end}"
        resp = s.get(url, timeout=5, proxies=proxies)
        json = resp.json()
        ##print(json)
        return json


class CZSCStrategy(bt.Strategy):
    state = "close"
    last = ""
    buy_price = 0
    max_price = 0

    def __init__(self):  # 初始化
        self.data_close = self.datas[0].close  # 指定价格序列
        # 初始化交易指令、买卖价格和手续费
        self.base_analyzer = CZSCAnalyzer(
            "btcusd",
            300,
            [
                300,
            ],
        )
        self.order = None
        self.signals = set()

    def next(self):  # 买卖策略
        dt = self.datas[0].datetime[0]
        o = self.datas[0].open[0]
        h = self.datas[0].high[0]
        l = self.datas[0].low[0]
        c = self.datas[0].close[0]
        v = self.datas[0].volume[0]
        self.base_analyzer.step(bt.num2date(dt), o, h, l, c, v)
        if Observer.sigal:
            t, bar, _ = Observer.sigal
            offset = NewBar.OBJS[-1].index - bar.index
            if offset > 2:
                return
            if Observer.sigal in self.signals:
                return
            self.signals.add(Observer.sigal)
            ##print(t, bar, bst)
        bi = []
        global global_bi
        for i in global_bi:
            tmp = i.strip().split(",")
            if len(bi) == 0 or bi[-1][0] != tmp[0]:
                bi.append(tmp)
            else:
                bi[-1] = tmp
        price = self.data.close[0]
        high = self.data.high[0]
        if self.position:
            self.max_price = max(self.max_price, high)
            if bi[-1][0] == "Up" and price < float(bi[-1][4]):
                self.sell(size=self.position.size)
                self.state = "stop"
            elif price < self.buy_price:
                self.sell(size=self.position.size)
                self.last = bi[-1][3]
                self.state = "close"
        elif self.state == "stop":
            if bi[-1][0] == "Down":
                self.state = "close"
            elif price > float(bi[-1][4]):
                cash = self.broker.getcash()
                self.buy(size=int(cash / price * 0.9))
                self.buy_price = price
                self.state = "open"
        elif len(bi) > 5 and price > float(bi[-1][4]):
            if self.module2(bi) and (price > self.max_price or self.last != bi[-1][3]):
                cash = self.broker.getcash()
                self.buy(size=int(cash / price * 0.9))
                self.buy_price = price
                self.max_price = high
                self.state = "open"
        Observer.sigal = None
        global global_state
        global_state = self.state
        return

    def module(self, bi: list[list]) -> bool:
        if len(bi) < 6 or bi[-1][0] == "Up":
            return False
        bi = bi[-6:]
        len_ = [abs(float(i[2]) - float(i[4])) for i in bi]
        result = self.cv_cal(len_[:3])
        if result > 0.2:
            return False
        result = self.cv_cal(len_[1:4])
        if result < 0.2:
            return False
        if (
            False
            # float(bi[4][4]) > float(bi[1][4])
            # or float(bi[4][4]) > float(bi[0][2])
            # or len_[5] > len_[3]
        ):
            return False
        return True

    def module2(self, bi: list[list]) -> bool:
        if len(bi) < 8 or bi[-1][0] == "Up":
            return False
        bi = bi[-8:]
        date_format = "%Y%m%d"
        time_len = [
            (
                datetime.datetime.strptime(i[3], date_format)
                - datetime.datetime.strptime(i[1], date_format)
            ).days
            for i in bi
        ]
        result = abs(float(bi[2][4]) / float(bi[0][4]) - 1)
        if result > 0.05:
            return False
        result = self.cv_cal(time_len[:3])
        if result > 0.5 or sum(time_len[:3]) > 110:
            return False
        len_ = [abs(float(i[2]) - float(i[4])) for i in bi]
        result = self.cv_cal(len_[:3])
        if result > 0.2 or float(bi[3][4]) < float(bi[0][2]):
            return False
        if float(bi[1][4]) < float(bi[3][4]):
            return False
        if (
            self.cv_cal(len_[4:8]) < 0.2
            or len_[4] > len_[3]
            or len_[4] > len_[5]
            or len_[6] > len_[5]
            or len_[6] > len_[7]
            # len_[4] + len_[6]
            # > len_[5]
            # or len_[5] > len_[3]
            # or len_[7] > len_[5]
        ):
            return False
        return True

    def cv_cal(self, float_list: list[float]):
        return np.std(float_list) / np.mean(float_list)

    def log(self, txt, dt=None, do_print=False):  # 日志函数
        dt = dt or bt.num2date(self.datas[0].datetime[0])

    # print("%s, %s" % (dt, txt))

    def notify_order(self, order):  # 记录交易执行情况
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Rejected:
            self.log(
                f"Rejected : order_ref:{order.ref}  data_name:{order.p.data._name}"
            )

        if order.status == order.Margin:
            self.log(f"Margin : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Cancelled:
            self.log(
                f"Concelled : order_ref:{order.ref}  data_name:{order.p.data._name}"
            )

        if order.status == order.Partial:
            self.log(f"Partial : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f" BUY : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}"
                )

            else:  # Sell
                self.log(
                    f" SELL : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}"
                )

    def notify_trade(self, trade):  # 记录交易收益情况
        if not trade.isclosed:
            return
        self.log(f"策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}")

    def stop(self):  # 回测结束后输出结果
        self.log(
            "期末总资金 %.2f" % (self.broker.getvalue()),
            do_print=True,
        )


def main_load_file1(path: str = "btcusd-300-1713295800-1715695500.dat"):
    obj = CZSCAnalyzer.load_file(path)
    obj.toCharts()
    return obj


def main_bitstamp(symbol="btcusd", limit=500, freq=Freq.m5):
    def func():
        bitstamp = Bitstamp(symbol, freq=freq)
        bitstamp.init(int(limit))
        bitstamp.toCharts()

    return func


# app = FastAPI()
# # priority_queue = asyncio.PriorityQueue()
# # queue = Observer.queue  # asyncio.Queue()
# app.mount(
#     "/charting_library",
#     StaticFiles(directory="charting_library"),
#     name="charting_library",
# )
# templates = Jinja2Templates(directory="templates")


async def process_queue():
    while True:
        message = await Observer.queue.get()
        try:
            await handle_message(message)
        except Exception as e:
            # print(f"Error handling message: {e}")
            traceback.print_exc()
        finally:
            Observer.queue.task_done()


# @app.on_event("startup")
# async def startup_event():
#     # 启动队列处理任务
#     asyncio.create_task(process_queue())


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data = await websocket.receive_text()
#             message = json.loads(data)
#             if message["type"] == "ready":
#                 exchange = message["exchange"]
#                 symbol = message["symbol"]
#                 freq = message["freq"]
#                 limit = message["limit"]
#                 # print(message)
#                 if Observer.thread is not None:
#                     tmp = Observer.thread
#                     Observer.thread = None
#                     Observer.sigals.clear()
#                     Observer.sigal = None
#                     RawBar.OBJS = []
#                     NewBar.OBJS = []
#                     FenXing.OBJS = []
#                     Bi.OBJS = []
#                     Bi.FAKE = None
#                     Duan.OBJS = []
#                     Duan.DUAN_OBJS = []
#                     ZhongShu.OBJS = []
#                     ZhongShu.BI_OBJS = []
#                     ZhongShu.DUAN_OBJS = []
#                     ZhongShu.DUAN_DUAN_OBJS = []
#                     tmp.join(1)
#                     time.sleep(1)

#                 Observer.thread = Thread(
#                     target=main_bitstamp(symbol=symbol, freq=freq, limit=limit)
#                 )  # 使用线程来运行main函数
#                 Observer.thread.start()
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)


# @app.get("/czsc")
# def static_czsc():
#     with open("czsc.html", "r") as f:
#         return HTMLResponse(f.read())


# @app.get("/")
# async def main_page(
#     request: Request,
#     nol: str = "network",
#     exchange: str = "bitstamp",
#     symbol: str = "btcusd",
#     step: int = 300,
#     limit: int = 500,
# ):
#     resolutions = {
#         60: "1",
#         180: "3",
#         300: "5",
#         900: "15",
#         1800: "30",
#         3600: "1H",
#         7200: "2H",
#         14400: "4H",
#         21600: "6H",
#         43200: "12H",
#         86400: "1D",
#         259200: "3D",
#     }
#     if resolutions.get(step) is None:
#         return resolutions

#     exchange = "bitstamp"

#     # if not (symbol in ("btcusd", )):
#     #    return resolutions
#     # print(request.base_url)
#     Observer.CAN = True
#     charting_library = str(
#         request.url_for("charting_library", path="/charting_library.standalone.js")
#     )
#     # print(charting_library)
#     return HTMLResponse(
#         """<!DOCTYPE html>
# <html lang="zh">
# <head>
#     <title>TradingView Chart with WebSocket</title>
#     <meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,minimum-scale=1.0">
#     <script type="text/javascript" src="$charting_library$"></script>
#     <script type="text/javascript">
#         const shape_ids = new Array(); // id 映射
#         const debug = false;
#         const exchange = "$exchange$";
#         const ticker = "$symbol$";
#         const name = ticker;//"BTCUSD"
#         const description = ticker;//"Bitcoin/USD"
#         const interval = "$interval$";
#         const step = "$step$";
#         const limit = "$limit$";
#         const socket = new WebSocket("ws://localhost:8080/ws");

#         socket.onopen = () => {
#             console.log("WebSocket connection established");
#             socket.send(JSON.stringify({
#                 type: "ready",
#                 exchange: exchange,
#                 symbol: name,
#                 freq: step,
#                 limit: limit
#             }));
#         };

#         socket.onclose = () => {
#             console.log("WebSocket connection closed");
#         };

#         socket.onerror = (error) => {
#             console.error("WebSocket error:", error);
#         };

#         let datafeed = {
#             onReady: (callback) => {
#                 console.log("[Datafeed.onReady]: Method call");
#                 setTimeout(() => callback({
#                     supports_search: false,
#                     supports_group_request: false,
#                     supports_marks: false,
#                     supports_timescale_marks: true,
#                     supports_time: true,
#                     supported_resolutions: [interval,],//["1s", "1", "3", "5", "6", "12", "24", "30", "48", "64", "128", "1H", "2H", "3H", "4H", "6H", "8H", "12H", "36H", "1D", "2D", "3D", "5D", "12D", "1W"],
#                 }));
#             },

#             searchSymbols: async (
#                 userInput,
#                 exchange,
#                 symbolType,
#                 onResultReadyCallback,
#             ) => {
#                 console.log("[Datafeed.searchSymbols]: Method call", userInput, exchange, symbolType);

#             },

#             resolveSymbol: async (
#                 symbolName,
#                 onSymbolResolvedCallback,
#                 onResolveErrorCallback,
#                 extension
#             ) => {
#                 console.log("[Datafeed.resolveSymbol]: Method call", symbolName);
#                 //return ;
#                 const symbolInfo = {
#                     exchange: exchange,
#                     ticker: ticker,
#                     name: name,
#                     description: description,
#                     type: "",
#                     session: "24x7",
#                     timezone: "Asia/Shanghai",
#                     minmov: 1,
#                     pricescale: 100000000, // 精度 数值越高小数点
#                     visible_plots_set: "ohlcv",
#                     has_no_volume: true,
#                     has_weekly_and_monthly: false, // 周线 月线
#                     supported_resolutions: ["1", "3", "5", "15", "30", "1H", "2H", "4H", "6H", "12H", "1D", "3D"],
#                     volume_precision: 1,
#                     data_status: "streaming",
#                     has_intraday: true,
#                     //intraday_multipliers: [5,], //["1", "3", "5", "15", "30", "60", "120", "240", "360", "720"],
#                     has_seconds: false,
#                     //seconds_multipliers: ["1S",],
#                     has_daily: true,
#                     //daily_multipliers: ["1", "3"],
#                     has_ticks: true,
#                     monthly_multipliers: [],
#                     weekly_multipliers: [],
#                 };
#                 try {
#                     onSymbolResolvedCallback(symbolInfo);
#                 } catch (err) {
#                     onResolveErrorCallback(err.message);
#                 }

#             },

#             getBars: async (
#                 symbolInfo,
#                 resolution,
#                 periodParams,
#                 onHistoryCallback,
#                 onErrorCallback,
#             ) => {
#                 const {from, to, firstDataRequest} = periodParams;
#                 console.log("[Datafeed.getBars]: Method call", symbolInfo, resolution, from, to, firstDataRequest);
#                 try {
#                     onHistoryCallback([], {noData: true});

#                 } catch (error) {
#                     console.log("[Datafeed.getBars]: Get error", error);
#                     onErrorCallback(error);
#                 }
#             },

#             subscribeBars: (
#                 symbolInfo,
#                 resolution,
#                 onRealtimeCallback,
#                 subscriberUID,
#                 onResetCacheNeededCallback,
#             ) => {
#                 console.log(
#                     "[Datafeed.subscribeBars]: Method call with subscriberUID:",
#                     symbolInfo,
#                     resolution,
#                     subscriberUID,
#                 );
#                 socket.onmessage = function (event) {
#                     const message = JSON.parse(event.data);
#                     if (debug) console.info(message);
#                     if (message.type === "realtime") {
#                         const bar = {
#                             time: new Date(message.timestamp).getTime(), // Unix timestamp in milliseconds
#                             close: message.close,
#                             open: message.open,
#                             high: message.high,
#                             low: message.low,
#                             volume: message.volume,
#                         };
#                         onRealtimeCallback(bar);
#                         //createShape(message.shape);
#                     } else if (message.type === "shape") {
#                         if (message.cmd === "bi_append" || message.cmd === "duan_append" || message.cmd === "zs_append" || message.cmd === "feature_append") {
#                             addShapeToChart(message);
#                         } else if (message.cmd === "bi_remove" || message.cmd === "duan_remove" || message.cmd === "zs_remove" || message.cmd === "feature_remove") {
#                             delShapeById(message)
#                         } else if (message.cmd === "bi_modify" || message.cmd === "duan_modify" || message.cmd === "zs_modify" || message.cmd === "feature_modify") {
#                             modifyShape(message)
#                         }
#                     } else {
#                         console.log("未知消息", message);
#                     }
#                 };
#             },

#             unsubscribeBars: (subscriberUID) => {
#                 console.log(
#                     "[Datafeed.unsubscribeBars]: Method call with subscriberUID:",
#                     subscriberUID,
#                 );
#                 socket.close();
#             }
#         };


#         function addShapeToChart(obj) {
#             if (window.tvWidget) {
#                 const shape_id = window.tvWidget.chart().createMultipointShape(obj.points, obj.options);
#                 shape_ids [obj.id] = shape_id;
#                 const shape = window.tvWidget.chart().getShapeById(shape_id);
#                 shape.setProperties(obj.properties);
#                 shape.bringToFront();
#                 //console.log(obj.id, shape_id);
#                 //console.log("add", obj.name, obj.id);
#             }
#         }

#         function delShapeById(obj) {
#             if (window.tvWidget) {
#                 try {
#                     const id = shape_ids[obj.id];
#                     delete shape_ids[obj.id];
#                     const shape = window.tvWidget.chart().getShapeById(id);
#                     if (debug) console.log(id, shape);
#                     window.tvWidget.chart().removeEntity(id);
#                     //console.log("del", shapeId, id);
#                 } catch (e) {
#                     console.log("删除失败", obj, e)
#                 }

#             }
#         }

#         function createShape(obj) {
#             if (window.tvWidget) {
#                 const shape_id = window.tvWidget.chart().createShape(obj.point, obj.options);
#                 shape_ids [obj.id] = shape_id;
#                 const shape = window.tvWidget.chart().getShapeById(shape_id);
#                 shape.bringToFront();
#                 //shape.setProperties(obj.options);
#             }
#         }

#         function modifyShape(obj) {
#             const id = shape_ids[obj.id];
#             try {
#                 const shape = window.tvWidget.chart().getShapeById(id);
#                 if (shape) {
#                     if (debug) console.log(obj);
#                     //console.log(shape.getProperties());
#                     shape.setPoints(obj.points);
#                     shape.setProperties(obj.properties);
#                     shape.bringToFront();

#                 } else {
#                     console.log("Shape does not exist.");
#                 }
#             } catch (e) {
#                 console.log("修改失败", id, obj, e)
#             }
#         }


#         function initOnReady() {
#             //console.log("init widget");
#             const widget = (window.tvWidget = new TradingView.widget({
#                 symbol: exchange + ":" + description, // Default symbol
#                 interval: interval, // Default interval
#                 timezone: "Asia/Shanghai",
#                 fullscreen: true, // Displays the chart in the fullscreen mode
#                 container: "tv_chart_container", // Reference to an attribute of the DOM element
#                 datafeed: datafeed,
#                 library_path: "charting_library/",
#                 locale: "zh",
#                 theme: "dark",
#                 debug: false,
#                 timeframe: "3D",
#                 user_id: "public_user_id",
#                 client_id: "yourserver.com",
#                 favorites: {
#                     intervals: ["1", "3", "5"],
#                     drawingTools: ["LineToolPath", "LineToolRectangle", "LineToolTrendLine"],
#                 },
#                 disabled_features: [
#                     "use_localstorage_for_settings", // 本地设置
#                     "header_symbol_search", // 搜索
#                     "header_undo_redo", // 重做
#                     "header_screenshot", // 截图
#                     //"header_resolutions",// 周期
#                     "header_compare", // 对比叠加
#                     "header_chart_type",
#                     "go_to_date", // 日期跳转
#                 ],
#                 time_frames: [
#                     {text: "3d", resolution: "5", description: "3 Days"},
#                     {text: "7d", resolution: "5", description: "7 Days"},
#                 ],
#             }));
#             widget.headerReady().then(function () {
#                 //widget.activeChart().createStudy("MACD");

#                 function createHeaderButton(text, title, clickHandler, options) {
#                     const button = widget.createButton(options);
#                     button.setAttribute("title", title);
#                     button.textContent = text;
#                     button.addEventListener("click", clickHandler);
#                 }

#                 createHeaderButton("笔买卖点", "显示隐藏买卖点", function () {
#                     widget.activeChart().getAllShapes().forEach(({name, id}) => {
#                         if (name === "arrow_up" || name === "arrow_down") {
#                             const shape = window.tvWidget.chart().getShapeById(id);
#                             const properties = shape.getProperties();
#                             if (properties.title === "BiFS" || properties.title === "BiSS" || properties.title === "BiTS"
#                                 || properties.title === "BiFB" || properties.title === "BiSB" || properties.title === "BiTB"
#                             )
#                                 shape.setProperties({visible: !properties.visible})
#                         }
#                     });
#                 });

#                 createHeaderButton("段买卖点", "显示隐藏买卖点", function () {
#                     widget.activeChart().getAllShapes().forEach(({name, id}) => {
#                         if (name === "arrow_up" || name === "arrow_down") {
#                             const shape = window.tvWidget.chart().getShapeById(id);
#                             const properties = shape.getProperties();
#                             if (properties.title === "DuanFS" || properties.title === "DuanSS" || properties.title === "DuanTS"
#                                 || properties.title === "DuanFB" || properties.title === "DuanSB" || properties.title === "DuanTB"
#                             )
#                                 shape.setProperties({visible: !properties.visible})
#                         }
#                     });
#                 });

#                 createHeaderButton("特征序列", "显示隐藏特征序列", function () {
#                     widget.activeChart().getAllShapes().forEach(({name, id}) => {
#                         if (name === "trend_line") {
#                             const shape = window.tvWidget.chart().getShapeById(id);
#                             const properties = shape.getProperties();
#                             if (properties.text.indexOf("feature") === 0)
#                                 shape.setProperties({visible: !properties.visible})
#                         }
#                     });
#                 });

#                 createHeaderButton("笔", "显示隐藏笔", function () {
#                     widget.activeChart().getAllShapes().forEach(({name, id}) => {
#                         if (name === "trend_line") {
#                             const shape = window.tvWidget.chart().getShapeById(id);
#                             const properties = shape.getProperties();
#                             if (properties.text.indexOf("bi") === 0)
#                                 shape.setProperties({visible: !properties.visible})
#                         }
#                     });
#                 });
#                 createHeaderButton("段", "显示隐藏段", function () {
#                     widget.activeChart().getAllShapes().forEach(({name, id}) => {
#                         if (name === "trend_line") {
#                             const shape = window.tvWidget.chart().getShapeById(id);
#                             const properties = shape.getProperties();
#                             if (properties.text.indexOf("duan") === 0)
#                                 shape.setProperties({visible: !properties.visible})
#                         }
#                     });
#                 });
#                 createHeaderButton("笔中枢", "显示隐藏笔中枢", function () {
#                     widget.activeChart().getAllShapes().forEach(({name, id}) => {
#                         if (name === "rectangle") {
#                             const shape = window.tvWidget.chart().getShapeById(id);
#                             const properties = shape.getProperties();
#                             if (properties.text.indexOf("Bizs") === 0)
#                                 shape.setProperties({visible: !properties.visible})
#                         }
#                     });
#                 });
#                 createHeaderButton("段中枢", "显示隐藏段中枢", function () {
#                     widget.activeChart().getAllShapes().forEach(({name, id}) => {
#                         if (name === "rectangle") {
#                             const shape = window.tvWidget.chart().getShapeById(id);
#                             const properties = shape.getProperties();
#                             if (properties.text.indexOf("Duanzs") === 0)
#                                 shape.setProperties({visible: !properties.visible})
#                         }
#                     });
#                 });
#                 widget.onChartReady(function () {
#                     // https://www.tradingview.com/charting-library-docs/v26/api/interfaces/Charting_Library.SubscribeEventsMap/
#                     widget.subscribe("onTimescaleMarkClick", function (clientX, clientY, pageX, pageY, screenX, screenY) {
#                         console.log("[onTimescaleMarkClick]", clientX, clientY, pageX, pageY, screenX, screenY)
#                     })
#                     widget.subscribe("drawing_event", function (sourceId, drawingEventType) {
#                         // properties_changed, remove, points_changed, click
#                         if (debug) console.log("[drawing_event]", "id:", sourceId, "event type:", drawingEventType)
#                         if (drawingEventType.indexOf("click") === 0) {
#                             const shape = widget.activeChart().getShapeById(sourceId);
#                             const properties = shape.getProperties();
#                             const points = shape.getPoints();
#                             const toolname = shape._source.toolname;
#                             if (toolname === "LineToolTrendLine") {
#                                 shape.setProperties({showLabel: !properties.showLabel})
#                             }
#                             console.log(toolname, points, properties);
#                         }
#                     })
#                 });
#             });
#         }

#         window.addEventListener("DOMContentLoaded", initOnReady, false);
#     </script>
# </head>
# <body style="margin:0px;">
# <div id="tv_chart_container"></div>
# </body>
# </html>
# """.replace(
#             "$charting_library$", charting_library
#         )
#         .replace("$exchange$", exchange)
#         .replace("$symbol$", symbol)
#         .replace("$interval$", resolutions.get(step))
#         .replace("$limit$", str(limit))
#         .replace("$step$", str(step))
#     )


async def handle_message(message: dict):
    if message["type"] == "realtime":
        await manager.send_message(json.dumps(message))
    elif message["type"] == "shape":
        options = message["options"]
        # options["disableUndo"]= True
        properties = message["properties"]
        properties["frozen"] = True
        await manager.send_message(
            json.dumps(
                {
                    "type": "shape",
                    "name": message["name"],
                    "points": message["points"],
                    "id": message["id"],
                    "cmd": message["cmd"],
                    "options": options,
                    "properties": message["properties"],
                }
            )
        )
    elif message["type"] == "heartbeat":
        await manager.send_message(
            json.dumps(
                {
                    "type": "heartbeat",
                    "timestamp": message["timestamp"],
                }
            )
        )
    else:
        await manager.send_message(
            json.dumps({"type": "error", "message": "Unknown command type"})
        )


def synchronous_handle_message(message):
    # 向优先级队列中添加任务
    loop = asyncio.get_event_loop()
    loop.call_soon_threadsafe(Observer.queue.put_nowait, message)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


def huice():
    cerebro = bt.Cerebro()
    with open("btcusd-300-1713295800-1715695500.dat", "rb") as f:
        cerebro.adddata(
            bt.feeds.PandasData(
                dataname=bs2df(f.read()), timeframe=bt.TimeFrame.Minutes
            )
        )
        cerebro.addstrategy(CZSCStrategy)
        cerebro.broker.setcash(1000000)
        cerebro.broker.setcommission(commission=0.0005)
        # cerebro.addsizer(bt.sizers.FixedSize, stake=100)
        cerebro.run(runonce=False)
        end_value = cerebro.broker.getvalue()
    # print("期末总资金: %.2f" % end_value)


class MyCSVData(GenericCSVData):
    # 定义列名映射到Backtrader字段
    lines = ("datetime", "open", "close", "high", "low", "volume")
    params = (
        ("dtformat", "%Y-%m-%d"),  # 日期格式
        ("datetime", 0),  # 日期列索引
        ("open", 1),  # 开盘价列索引
        ("close", 2),  # 收盘价列索引
        ("high", 3),  # 最高价列索引
        ("low", 4),  # 最低价列索引
        ("volume", 5),  # 成交量列索引
        ("openinterest", -1),  # 开仓兴趣列索引，如果不存在可以设置为-1
    )


# RawBar.PATCHES[ts2int("2024-04-17 21:20:00")] = Pillar(62356, 62100)
manager = ConnectionManager()
Observer.TIME = 0.02


def main(code, from_date, to_date):
    global global_bi
    global_bi = []
    with open("bi.txt", "w") as f:
        pass
    # bit = main_load_file("btcusd-300-1713295800-1715695500.dat")
    cerebro = bt.Cerebro()
    # with open("btcusd-300-1713295800-1715695500.dat", "rb") as f:
    #     cerebro.adddata(
    #         bt.feeds.PandasData(
    #             dataname=bs2df(f.read()), timeframe=bt.TimeFrame.Minutes
    #         )
    #     )
    # 加载数据

    fromdate = datetime.datetime.strptime(from_date, "%Y%m%d")
    todate = datetime.datetime.strptime(to_date, "%Y%m%d")
    data = MyCSVData(
        dataname="stock_data.csv",
        fromdate=fromdate,
        todate=todate,
    )

    # 添加数据到回测引擎
    cerebro.adddata(data)
    cerebro.addstrategy(CZSCStrategy)
    start_cash = 10000
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.0005)
    # cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    cerebro.run(runonce=False)
    end_value = cerebro.broker.getvalue()
    # print("期末总资金: %.2f" % end_value)
    tq07_corUp, tq07_corDown = ["#E1440F", "#B0F76D"]
    tq_ksty07 = dict(
        volup=tq07_corUp, voldown=tq07_corDown, barup=tq07_corUp, bardown=tq07_corDown
    )
    # cerebro.plot(style="candle", **tq_ksty07)
    # print(len(Observer.sigals))

    # 计算总收益率
    total_return_percentage = ((end_value - start_cash) / start_cash) * 100

    # 假设回测周期是从date1到date2，计算总天数
    total_days = (todate - fromdate).days
    # 假设一年有365天，计算平均年化收益率
    annualized_return_percentage = (
        (1 + total_return_percentage / 100) ** (365 / total_days)
    ) - 1
    annualized_return_percentage *= 100  # 转换为百分比形式
    print(f"Average Annualized Return Percentage: {annualized_return_percentage:.2f}%")

    matplotlib.use("agg")
    if abs(annualized_return_percentage) > 0.1:
        figs = cerebro.plot(style="candle", **tq_ksty07)
        fig = figs[0][0]
        fig.savefig("photo/" + code + ".png")
    return annualized_return_percentage


def main2(code, from_date, to_date):
    global global_bi
    global global_state
    global_bi = []
    global_state = ""
    with open("bi.txt", "w") as f:
        pass
    # bit = main_load_file("btcusd-300-1713295800-1715695500.dat")
    cerebro = bt.Cerebro()
    # with open("btcusd-300-1713295800-1715695500.dat", "rb") as f:
    #     cerebro.adddata(
    #         bt.feeds.PandasData(
    #             dataname=bs2df(f.read()), timeframe=bt.TimeFrame.Minutes
    #         )
    #     )
    # 加载数据

    fromdate = datetime.datetime.strptime(from_date, "%Y%m%d")
    todate = datetime.datetime.strptime(to_date, "%Y%m%d")
    data = MyCSVData(
        dataname="stock_data.csv",
        fromdate=fromdate,
        todate=todate,
    )

    # 添加数据到回测引擎
    cerebro.adddata(data)
    cerebro.addstrategy(CZSCStrategy)
    start_cash = 10000
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.0005)
    # cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    cerebro.run(runonce=False)
    end_value = cerebro.broker.getvalue()
    # print("期末总资金: %.2f" % end_value)
    # cerebro.plot(style="candle", **tq_ksty07)
    # print(len(Observer.sigals))

    # 计算总收益率
    total_return_percentage = ((end_value - start_cash) / start_cash) * 100

    # 假设回测周期是从date1到date2，计算总天数
    total_days = (todate - fromdate).days
    # 假设一年有365天，计算平均年化收益率
    annualized_return_percentage = (
        (1 + total_return_percentage / 100) ** (365 / total_days)
    ) - 1
    annualized_return_percentage *= 100  # 转换为百分比形式
    return global_state


if __name__ == "__main__":
    main("605376", "20230830", "20240830")
