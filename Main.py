# coding=utf-8
import os
import sys
import io
import threading

import time
from typing import Sequence, List, Dict, Tuple, Optional, Union
import win32api
import cv2
import numpy
import pynput
import win32gui, win32ui, win32con

from VideoWriter import VideoWriter

"""屏蔽 pygame 的输出"""
_ = sys.stdout
sys.stdout = io.StringIO()
import pygame

sys.stdout = _
del _


def on_key_press_function(waitKey):
    """
    创建一个用于等待 waitKey 按下的函数（用于 pynput.keyboard.Listener 的 on_press 参数）
    :param waitKey:
    :return:
    """

    def on_press(key):
        try:
            return key.char != waitKey
        except Exception:
            return True

    return on_press


def inflateRect(rect: pygame.Rect, padding: int) -> pygame.Rect:
    """扩展矩形，中心不变，原 Rect 不变"""
    return rect.inflate(padding, padding)


def integrateRange(charWidth: int, k: float, ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    合并一维范围（合并互相包含的范围；将过小的缝隙（小于角色宽度*系数k）去除（合并相邻范围））
    """

    def _1in2(range1: Tuple[int, int], range2: Tuple[int, int]):
        """范围2是否包含范围1"""
        return range2[0] <= range1[0] <= range1[1] <= range2[1]

    sortedRanges: List[Tuple[int, int]] = list(sorted(ranges, key=lambda a: a[0]))  # 通过范围左端进行排序
    lastRange: Tuple[int, int] = tuple()
    pop = []
    for index, range_ in enumerate(tuple(sortedRanges)):  # tuple(): 防止后来为 sortedRanges 增加元素而报错
        if lastRange:
            if _1in2(range_, lastRange):  # range_ 左端比 lastRange 左端更靠右，因此只能是 lastRange 包含 range_
                # 若为包含关系，则去除内层矩形，即 range_
                if index not in pop:  # 防止重复删除
                    pop.append(index)
        lastRange = range_
    while pop:  # 清空 pop 并清除其指向的元素
        i = pop.pop(-1)  # 倒序弹出，防止改变了元素序号
        sortedRanges.pop(i)
    # sortedRanges 仍有序
    lastRange = tuple()
    newRanges = []
    for index, range_ in enumerate(sortedRanges):  # tuple(): 防止后来为 sortedRanges 增加元素而报错
        if lastRange and (range_[0] - lastRange[1]) < charWidth * k:  # 左右范围缝隙 < (角色宽度 * 系数)
            # 合并
            lastInsert = newRanges.pop(-1)  # 弹出上一个
            integrated = (lastRange[0], range_[1])  # 一次整合
            if (range_[0] - lastInsert[1]) < charWidth * k:  # 如果与上一个范围靠近或交叉
                newRanges.append((lastInsert[0], integrated[1]))  # 插入二次整合
            else:
                newRanges.append(integrated)  # 插入一次整合
        else:
            newRanges.append(range_)  # 无需整合
        lastRange = range_
    return newRanges


def findContours(img: numpy.ndarray) -> numpy.ndarray:
    """寻找图像中的边框"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getCaptureRange():
    """
    获得应该捕获的区域
    :return: 相对于左上点的偏移点(Tuple[int, int]), 捕获尺寸(Tuple[int, int])
    """
    return (0, 0), ScreenCapture.capture().shape[:2][::-1]


def pointsToRect(points: Sequence[Tuple[int, int]]):
    """将 python 对象表示的矩形转换为 pygame.Rect"""
    right = max(points, key=lambda a: a[0])
    left = min(points, key=lambda a: a[0])
    top = min(points, key=lambda a: a[1])
    bottom = max(points, key=lambda a: a[1])
    return pygame.Rect([left[0], top[1], right[0] - left[0], bottom[1] - top[1]])


class GameControl:
    controller = pynput.keyboard.Controller()
    tapTimeSep_s = 150 * 0.001
    direction: Optional[int] = None
    direction_dict_ch = {-2: "左二级", -1: "左一级", 0: "正向下", 1: "右一级", 2: "右二级", None: "停下"}
    direction_dict_en = {-2: "Left__2", -1: "Left__1", 0: "Down", 1: "Right_1", 2: "Right_2", None: "Ceased"}

    @classmethod
    def detectLoaded(cls) -> bool:
        """检测页面是否加载完毕"""
        img = ScreenCapture.capture()
        height, width = img.shape[:2]
        for y in range(height):
            for x in range(width):
                point = img[y, x]
                if point[0] == 10 and point[1] == 128 and point[2] == 255:  # 上部心形颜色
                    return True
        return False

    @classmethod
    def sleep(cls):
        time.sleep(cls.tapTimeSep_s)

    @classmethod
    def getDirectionInText(cls):
        """获得现在方向的英文解释"""
        return cls.direction_dict_en.get(cls.direction)

    @classmethod
    def changeDirection(cls, direction: Optional[int]):
        """
        *操作时请勿移动鼠标*
        通过控制键盘改变角色移动方向
        :param direction: 见 cls.direction_dict
        """
        if direction is None:
            cls.controller.tap("w")
        elif direction == -2:
            cls.controller.tap('a')
            cls.sleep()
            cls.controller.tap('a')
        elif direction == -1:
            cls.controller.tap('a')
        elif direction == -0:
            cls.controller.tap('s')
        elif direction == 1:
            cls.controller.tap('d')
        elif direction == 2:
            cls.controller.tap('d')
            cls.sleep()
            cls.controller.tap('d')
        else:
            raise ValueError("Wrong direction.")
        cls.direction = direction

    @classmethod
    def startGame(cls):
        """开始游戏"""
        cls.controller.tap(" ")

    @classmethod
    def refresh(cls):
        """刷新界面"""
        cls.controller.tap(pynput.keyboard.Key.f5)


class ScreenCapture:
    offset = (0, 0)  # 屏幕识别左上点
    captureSize = None  # 屏幕识别区域大小

    @classmethod
    def startPosition(cls, point: Tuple[int, int]):
        cls.offSet = point

    @classmethod
    def resetRange_offset8size(cls, offset: Tuple[int, int], size: Tuple[int, int]):
        """
        重设捕获区域
        :param offset: 左上偏移点
        :param size: 捕获尺寸
        """
        cls.offset = offset
        cls.captureSize = size

    @classmethod
    def resetRange_2point(cls, point_LeftTop: Tuple[int, int], point_RightBottom: Tuple[int, int]):
        """
        重设捕获区域
        :param point_LeftTop: 捕获区域左上点
        :param point_RightBottom: 捕获区域右下点
        """
        cls.offset = point_LeftTop
        cls.captureSize = tuple([j - i for i, j in zip(point_LeftTop, point_RightBottom)])

    @classmethod
    def capture(cls) -> numpy.ndarray:
        """
        捕获设定区域
        """
        # 此时 size 仍可为 None：不裁剪
        hwnd = 0  # 窗口的类名可以用Visual Studio的SPY++工具获取
        # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
        hwndDC = win32gui.GetWindowDC(hwnd)
        # 根据窗口的DC获取mfcDC
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        # mfcDC创建可兼容的DC
        saveDC = mfcDC.CreateCompatibleDC()
        # 创建bigmap准备保存图片
        saveBitMap = win32ui.CreateBitmap()
        # 获取监控器信息
        MoniterDev = win32api.EnumDisplayMonitors(None, None)
        w = MoniterDev[0][2][2]
        h = MoniterDev[0][2][3]
        # 为bitmap开辟空间
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        # 高度saveDC，将截图保存到saveBitmap中
        saveDC.SelectObject(saveBitMap)
        # 截取从左上角（0，0）长宽为（w，h）的图片
        saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
        tempPic = "temp.png"
        saveBitMap.SaveBitmapFile(saveDC, tempPic)
        im_opencv = cv2.imread(tempPic)
        cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2RGB)
        leftTopX, leftTopY = cls.offset
        size = cls.captureSize or im_opencv.shape[:2][::-1]  # 使用默认大小
        capWidth = size[0]
        capHeight = size[1]
        cut = im_opencv[leftTopY:leftTopY + capHeight, leftTopX:leftTopX + capWidth]
        # 内存释放
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        os.remove("temp.png")
        return cut


class CharacterFinder:
    """角色框寻找者"""
    # 最小储存帧数量（用于判断不动的矩形位置，见 search ）
    minStoreFrames = 4
    # 保存多帧中的矩形框位置
    rectFrames: List[Dict[Tuple[int, int], Sequence[Tuple[int, int]]]] = []

    @classmethod
    def ndArrayToTuple(cls, rectArrays: Sequence[numpy.ndarray]) -> List[List[Tuple[int, int]]]:
        """
        将 ndarray 中的值过滤出矩形边框 并转换为 python 对象
        :param rectArrays: 一帧中识别出的 rect 边框 ndarray
        :return:
        """
        # 当前处于一帧上下文
        # Tuple[int, int] 为坐标形式
        # List[Tuple[int, int]] 为一个 rect 的四角
        # List[List[Tuple[int, int]]] 为一个帧中所有 rect
        result: List[List[Tuple[int, int]]] = []
        for array in rectArrays:  # 进入一个矩形上下文
            if len(array) != 4:  # 过滤：非四个点，不是矩形
                continue
            if not array[0].any():  # 过滤：第一个点为(0, 0)，是最外层的框，去掉
                continue
            rect = []
            for subArray in array:  # 进入一个点上下文
                subArray = subArray[0]  # 数据特色（我也不知道为啥）
                x, y = subArray[0], subArray[1]
                rect.append((x, y))
            result.append(rect)
        return result

    @classmethod
    def storeFrame(cls, rectangles: Sequence[Sequence[Tuple[int, int]]]) -> bool:
        """
        储存识别到的矩形框
        :return: 是否建议进行寻找（search）
        """
        frameDict: Dict[Tuple[int, int], Sequence[Tuple[int, int]]] = {}  # 创建 rect 索引（第一个点为 key ）
        [frameDict.__setitem__(rect[0], rect) for rect in rectangles]
        cls.rectFrames.append(frameDict)
        return len(cls.rectFrames) >= cls.minStoreFrames

    @classmethod
    def search(cls, captureHeight: int) -> Optional[Sequence[Tuple[int, int]]]:
        """
        寻找 rectFrames 中不动的矩形框（角色矩形框，并返回）
        判断只会依据每个矩形框左上角的点（序列中第一项）
        :return: 返回角色框(Nullable)
        """
        still: Dict[Tuple[int, int], List[int]] = {}
        for frame in cls.rectFrames:
            [still.setdefault(rect, []).append(1) for rect in frame]  # 停留计数
        try:
            # charKeys: 角色矩形在 rectFrames 里字典的 key
            charKeys = filter(lambda key: len(still[key]) >= cls.minStoreFrames, still.keys())  # 判断停留点
            charRect = None  # 目标角色矩形
            for charKey in charKeys:
                rect = cls.rectFrames[0].get(charKey)  # rectFrames 每个字典都有此点，所以选第一个
                if captureHeight * 0.30 < rect[0][1] < captureHeight * 0.7:  # 查找符合条件的点（在屏幕中部）
                    charRect = rect
                    break
            return charRect
        except StopIteration:  # 无
            return None

    @classmethod
    def clearFrames(cls):
        cls.rectFrames.clear()


def drawRectangles(img: numpy.ndarray, rectangles: Sequence[Sequence[Tuple[int, int]]],
                   charRect: Sequence[Tuple[int, int]]):
    """
    在 img 上做上 rectangles 标记
    :param img:
    :param rectangles:
    :param charRect: 角色的矩形（用于特殊颜色标记）
    :return:
    """
    for rect in rectangles:
        for point in rect:
            if point in charRect:
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)
            cv2.drawMarker(img, point, color, markerSize=30)


def drawRanges(img: numpy.ndarray, *rangeLayers: Sequence[Tuple[int, int]],
               charRect: Sequence[Tuple[int, int]]):
    """
    在 img 上做上 ranges 标记
    :param img:
    :param rangeLayers: 范围（可多层）
    :param charRect: 角色的矩形（用于特殊颜色标记）
    :return:
    """
    drawY = max(charRect, key=lambda a: a[1])[1]  # 原范围标记高度，取角色矩形最低的高度
    delta = 20  # 每层范围标记增加的高度
    for point in charRect:
        cv2.drawMarker(img, point, (0, 255, 255), markerSize=30)
    for index, ranges in enumerate(rangeLayers):
        for range_ in ranges:
            cv2.line(img, (range_[0], drawY + index * delta), (range_[1], drawY + index * delta), (255, 0, 0),
                     thickness=3)


def test():
    # print(integrateRange(31, 1.5, [(225, 256), (243, 259), (239, 255), (336, 395), (272, 331), (432, 459), (266, 305)]))
    while True:
        print(GameControl.detectLoaded())


def main():
    # 初始化参数
    fps = 60  # 每秒读取帧数（不一定能达到）
    padding = 14  # 将矩形扩大该像素，防止误差
    directionLevel = 1  # 改变方向级数（1、2）
    ignoreGapFactor = 1.3  # 忽略缝隙系数（integrateRange 中的 k）
    tolerantFrames = 45  # 容错/容忍，允许角色丢失的最大帧数
    visibility = 10  # 视野，决定能看见角色前面多远的障碍（单位为角色高度）（不一定越大越好）
    offset, captureSize = getCaptureRange()  # 捕获区域的起始偏移点 和 捕获区域的尺寸（建议全屏且 offset = (0,0)）
    doRecordFrames = True  # 是否记录游戏情景
    output = "record.avi"  # 游戏情景的输出文件
    startRecordFrameNum = 90  # frameCount 达到 this 时开始记录游戏屏幕（获取大概帧率）
    fontThickness = 2  # 字厚度
    fontScale = 4  # 字大小
    fontColor = (255, 0, 0)  # 字色
    fontFace = cv2.FONT_HERSHEY_PLAIN  # 字体
    clock = pygame.time.Clock()
    ScreenCapture.resetRange_2point(offset, captureSize)  # 设置捕获区域

    print("打开新版 edge 浏览器，\n请打开 surf 并启用 surf 设置中的高可见性，\n启动全屏，\n然后敲击 G 键开始，\n点击 G 后请不要自己开始游戏，脚本会自动识别")
    with pynput.keyboard.Listener(
            on_press=on_key_press_function("g")
    ) as listener:
        listener.join()
    print("等待页面加载中")
    GameControl.refresh()  # 刷新 surf 界面
    while GameControl.detectLoaded():  # 检测游戏是否加载完毕
        clock.tick(fps)
    print("启动中")
    GameControl.sleep()  # 等待一会
    GameControl.startGame()  # 开始游戏
    print("游戏开始")
    while True:
        src = ScreenCapture.capture()
        contours = findContours(src)
        rectangles = CharacterFinder.ndArrayToTuple(contours)
        if CharacterFinder.storeFrame(rectangles):
            try:
                charRect = inflateRect(  # 扩展矩形
                    pointsToRect(  # 转换为 pygame.Rect
                        CharacterFinder.search(captureSize[1])),  # 搜索角色，参数为捕获高度
                    padding
                )
                break  # 直到获得charRect后停止
            except TypeError:
                print("获得角色失败")
            CharacterFinder.clearFrames()  # 清除所有帧，防止使用错误的帧
        clock.tick(fps)
    print("角色取得：", charRect)
    print()

    hasChar = True  # 角色是否仍在屏幕中（若为 False 则超出了忍耐帧数）
    lostFrames = 0  # 角色丢失帧数
    frameCount = 0  # 帧计数
    videoWriter = None  # type:VideoWriter  # 游戏情况记录者

    while hasChar:
        src = ScreenCapture.capture()
        contours = findContours(src)
        rectangles = CharacterFinder.ndArrayToTuple(contours)
        # 过滤无影响矩形后，将矩形变为一维范围，合并，去除狭窄缝隙，防止陷入死路或 “z走” 死循环。
        rectangles_Rect = [inflateRect(pointsToRect(r), padding) for r in rectangles]  # 转换为 pygame.Rect 表达，并扩大
        rectangles_xRange = [(r.left, r.right) for r in rectangles_Rect
                             if (charRect.centery + r.height * visibility >
                                 r.centery > charRect.centery)]  # 变为一维范围（过滤掉了无影响障碍与过远障碍（与视野有关））
        integratedRanges = integrateRange(charRect.width, ignoreGapFactor, rectangles_xRange)
        # print(charRect.width, ignoreGapFactor, rectangles_xRange,
        #       "->", sorted(integratedRanges, key=lambda a: a[0]))  # debug： 输出识别信息
        hasChar = False
        # 检测角色
        for rect in rectangles_Rect:
            # 检测角色是否仍存在
            if rect.topleft == charRect.topleft and rect.size == charRect.size:
                hasChar = True
                continue
        if not hasChar and tolerantFrames > lostFrames:  # 忍耐计数
            hasChar = True
            lostFrames += 1
            print(f"\r角色不在屏幕内，进度完成后将结束：{lostFrames / tolerantFrames:.2%}", end="")
        else:
            if lostFrames != 0:
                print(f"\r角色恢复")
                pass
            lostFrames = 0  # 重置丢失帧数

        # 若角色存在（严格判定，不容忍） 改变方向 并 保存图像
        if lostFrames == 0:
            changedDirection = False
            for range_ in integratedRanges:
                # 判断障碍是否对着角色矩形，改变行动方向
                if range_[0] <= charRect.left <= range_[1] or \
                        range_[0] <= charRect.right <= range_[1]:  # 角色一边（左或右或都）在障碍范围内，向出路最短
                    if charRect.centerx < (sum(range_) / 2):
                        GameControl.changeDirection(-directionLevel)
                    else:
                        GameControl.changeDirection(directionLevel)
                    changedDirection = True
                elif charRect.left <= range_[0] and range_[1] <= charRect.right:  # 障碍范围在角色内，向左（其他情况被上面包括）
                    GameControl.changeDirection(directionLevel)
                    changedDirection = True
                else:
                    GameControl.changeDirection(0)  # 向下
                # 改变了方向，不再改变
                if changedDirection:
                    break

            if startRecordFrameNum <= frameCount:
                if videoWriter is None:
                    # 游戏情景记录
                    # 通过前面的 clock.tick 得到了大概的fps
                    videoWriter = VideoWriter(output, *ScreenCapture.captureSize, round(clock.get_fps()))
                else:
                    # 记录捕获的游戏情景与识别出的信息
                    drawRanges(src, rectangles_xRange, *[[i] for i in integratedRanges],  # 展开成多行
                               charRect=[charRect.topleft, charRect.topright, charRect.bottomright,
                                         charRect.bottomleft])
                    text = GameControl.getDirectionInText()
                    cv2.putText(src, text, (100, 100), fontFace, fontScale, fontColor, thickness=fontThickness)
                    videoWriter.write(src)
        # 显示
        # out_img = src.copy()
        # drawRanges(out_img, rectangles_xRange, *[[i] for i in integratedRanges],  # 展开成多行
        #           charRect=[charRect.topleft, charRect.topright, charRect.bottomright, charRect.bottomleft])
        # cv2.namedWindow("debug", cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("debug", out_img)
        # cv2.waitKey(1)

        clock.tick(fps)  # 延迟，控制帧率
        frameCount += 1  # 帧计数

    print("角色丢失 100%")
    if doRecordFrames:
        print("保存图像中...")
        # 保存角色丢失时的图像 (只是关闭输出流)
        try:
            videoWriter.close()
        except AttributeError:
            pass
        print(f"图像保存完成 -> {os.path.abspath(output)}")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
