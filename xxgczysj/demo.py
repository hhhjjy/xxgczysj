# coding:utf-8
import sys

from PyQt5.QtCore import Qt, QUrl,QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QDesktopServices
from qfluentwidgets import FluentIcon as FIF
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5 import QtCore
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2
import traceback
from models.experimental import attempt_load
from utils.dataloaders import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size,  check_imshow,non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from qfluentwidgets import *

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_camera = pyqtSignal(np.ndarray)
    send_camera_raw = pyqtSignal(np.ndarray)
    send_mp4=pyqtSignal(np.ndarray)
    send_mp4_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            # device = select_device(device)
            # half &= device.type != 'cpu'  # half precision only supported on CUDA
            device=torch.device('cpu')
            # Load model
            model = attempt_load(self.weights,device=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # self.label=
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            while True:
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, device=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {names[name]: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                    if self.rate_check:
                        time.sleep(1/self.rate)
                    im0 = annotator.result()
                    self.result=im0
                    # self.send_img.emit(im0)
                    # self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                        if self.camera_detect_flag:
                            self.send_camera.emit(im0)
                        self.send_camera_raw.emit(im0s)
                    elif self.source.lower().endswith(('mp4')):
                        self.send_mp4_raw.emit(im0)
                        self.send_mp4.emit(im0s)
                    else:
                        self.send_img.emit(im0)
                    self.send_statistic.emit(statistic_dic)
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    if percent == self.percent_length:
                        print(count)
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break

        except Exception as e:
            print('{:*^60}'.format('直接打印出e, 输出错误具体原因'))
            print(e)
            print('{:*^60}'.format('使用repr打印出e, 带有错误类型'))
            print(repr(e))
            print('{:*^60}'.format('使用traceback的format_exc可以输出错误具体位置'))
            exstr = traceback.format_exc()
            print(exstr)
# class Window(SplitFluentWindow,object):
def restart_program():
    # 启动一个新进程
    process = QtCore.QProcess()
    # 获取当前应用程序的路径
    program = sys.argv[0]
    # 重启当前应用程序
    process.startDetached(program)
    # 退出当前应用程序
    QApplication.quit()

class Config(QConfig):
    musicFolders=ConfigItem("Folders", "LocalMusic", [], FolderListValidator())
    enableAcrylicBackground = ConfigItem("MainWindow", "EnableAcrylicBackground", False, BoolValidator())
    opacity = RangeConfigItem("Online", "PageSize", 100, RangeValidator(0, 100))
class Window(SplitFluentWindow):
    def __init__(self):
        super().__init__()
        self.resize(1350, 780)
        self.setWindowTitle('信息工程专业设计——基于YOLOv5的无人机目标检测')
        # self.touxiang_path=r'D:\Now\信息工程专业设计\PyQt5-YOLOv5-7.0-2\clock\resource\images\头像.jpg'
        self.touxiang_path=os.path.join(os.getcwd(), 'images','头像.jpg')

        # self.setWindowIcon(QIcon(':/qfluentwidgets/images/logo.png'))
        self.setWindowIcon(QIcon(self.touxiang_path))
        # self.setWindowIcon(QIcon(r'.\images\头像.jpg'))

        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QtCore.QSize(102, 102))
        # self.background_path=r"D:\Now\信息工程专业设计\PyQt5-YOLOv5-7.0-2\clock\resource\images\粉.png"
        # self.background_path = r".\images\粉.png"
        self.background_path = os.path.join(os.getcwd(), 'images', '粉.png')
        self.show()
        self.setupUi()
        self.setBackground()
        loop = QtCore.QEventLoop(self)
        QTimer.singleShot(1000, loop.quit)
        loop.exec()

        self.splashScreen.finish()
        self.setWindowOpacity(1)

    def setBackground(self):

        self.palette = QtGui.QPalette()
        window_size = self.size()

        # 调整图片大小以适应窗口尺寸
        # self.background_path=r"D:\Pictures\Saved Pictures\2024-10-22 10_50_10.png"
        self.background =QPixmap(self.background_path)
        self.background = self.background.scaled(window_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.palette.setBrush(QtGui.QPalette.Background, QtGui.QBrush(self.background))
        self.setPalette(self.palette)

    def resizeEvent(self, e):
        try:
            super().resizeEvent(e)
            self.background = QPixmap(self.background_path).scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.palette.setBrush(QtGui.QPalette.Background, QtGui.QBrush(self.background))
            self.setPalette(self.palette)
        except:
            pass

    def setupUi(self):
        self.ptpath = os.path.join(os.getcwd(), 'pt')
        self.savepath = os.path.join(os.getcwd(), 'save')
        # self.touxiang_path = os.path.join(os.getcwd(), 'images','头像.jpg')
        self.addtab()
        self.addtab4()
        self.addtab3()
        self.addtab2()
        self.retranslateUi()
        self.initNavigation()
        self.initWindow()
        self.m_flag = False
        self.try_all_model = False
        self.camera_open = 1

        self.pt_list = []

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(3000)

        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.det_thread.label))
        self.det_thread.send_camera.connect(lambda x: self.show_image(x, self.camera1))
        self.det_thread.send_camera_raw.connect(lambda x: self.show_image(x, self.camera))
        self.det_thread.send_mp4.connect(lambda x: self.show_image(x, self.label))
        self.det_thread.send_mp4_raw.connect(lambda x: self.show_image(x, self.label_2))
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))

        self.fileButton.clicked.connect(self.open_file)

        self.stopButton.clicked.connect(self.stop)

        self.jiance1.clicked.connect(lambda: self.run_or_continue(self.img1, self.comboBox_3.currentText()))
        self.jiance2.clicked.connect(lambda: self.run_or_continue(self.img2, self.comboBox_4.currentText()))
        self.jiance3.clicked.connect(lambda: self.run_or_continue(self.img3, self.comboBox_5.currentText()))
        self.baocun1.clicked.connect(lambda: self.saveimage(self.img1))
        self.baocun2.clicked.connect(lambda: self.saveimage(self.img2))
        self.baocun3.clicked.connect(lambda: self.saveimage(self.img3))

        self.checkBox_2.clicked.connect(self.tryallmodel)

        self.comboBox.currentTextChanged.connect(self.change_model)

        # 摄像头
        self.pushButton.clicked.connect(lambda checked: self.tog_camera(checked))
        self.pushButton_10.clicked.connect(lambda checked: self.tog_det(checked))
        self.det_thread.camera_detect_flag = False

        # 视频
        self.pushButton_13.clicked.connect(self.open_file)
        # self.pushButton_6.clicked.connect(lambda:self.run_or_continue(self.label,self.comboBox_6.currentText()))

        # rtsp
        self.pushButton_14.clicked.connect(self.close_cam)
        self.pushButton_8.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(lambda: self.run_or_continue1(self.label, self.comboBox_6.currentText()))
        # self.checkBox.clicked.connect(self.checkrate)
        # self.saveCheckBox.clicked.connect(self.is_save)

        # setting
        # self.label_9.setText(self.ptpath)
        # self.label_11.setText(self.savepath)
        # self.pushButton_16.clicked.connect(self.open_PTfile)
        # self.pushButton_17.clicked.connect(self.chose_savepath)
        self.ptcard.button.clicked.connect(self.open_PTfile)
        self.savecard.button.clicked.connect(self.chose_savepath)
        # self.load_setting()
        self.click_list = []
        # mainWindow.setCentralWidget(self.centralwidget)
        # QtCore.QMetaObject.connectSlotsByName(mainWindow)


    def addtab4(self):
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.cfg=Config()
        self.expandLayout = ExpandLayout(self.tab_4)
        self.musicInThisPCGroup = SettingCardGroup(
            self.tr(""), self.tab_4)
        self.ptcard = PushSettingCard(
            text="选择文件夹",
            icon=FluentIcon.DOWNLOAD,
            title="模型目录",
            content=self.ptpath,
            # clicked=self.open_PTfile
        )
        self.savecard = PushSettingCard(
            text="选择文件夹",
            icon=FluentIcon.SAVE,
            title="保存目录",
            content=self.savepath,
            # clicked=self.chose_savepath
        )
        self.opacitycard = RangeSettingCard(
            self.cfg.opacity,
            title="窗口透明度",
            icon=FluentIcon.HIDE,
            content="小心不要把窗口调没了😉",
        )
        self.backgroundcard = PushSettingCard(
            text="选择背景图片",
            icon=FluentIcon.PHOTO,
            title="背景图片",
            content=self.background_path,
            # clicked=self.chose_savepath
        )
        self.touxiangcard = PushSettingCard(
            text="选择头像",
            icon=FluentIcon.PHOTO,
            title="头像",
            content=self.touxiang_path,
            # clicked=self.chose_savepath
        )
        # self.themeColorCard=CustomColorSettingCard(
        #     self.cfg.opacity,
        #     FIF.PALETTE,
        #     self.tr('背景主题'),
        #     self.tr('改变背景'),
        #     self.musicInThisPCGroup
        # )
        # # self.themeColorCard.defaultRadioButton.setText('蓝色')
        # self.themeColorCard.defaultRadioButton = RadioButton(
        #     self.tr('蓝'), self.themeColorCard.radioWidget)
        # self.theme_blue='D:\Pictures\Saved Pictures\黑.png'
        # self.themeColorCard.customRadioButton = RadioButton(
        #     self.tr('粉'), self.themeColorCard.radioWidget)
        # self.theme_fen='D:\Pictures\Saved Pictures\粉.png'
        # self.themeColorCard.customRadioButton = RadioButton(
        #     self.tr('红'), self.themeColorCard.radioWidget)
        # self.theme_hong='D:\Pictures\Saved Pictures\黑.png'

        # self.themeColorCard.customLabel = QtWidgets.QLabel(
        #     self.tr('Custom color'), self.themeColorCard.customColorWidget)
        # self.themeColorCard.chooseColorButton = QtWidgets.QPushButton(
        #     self.tr('Choose color'), self.themeColorCard.customColorWidget)


        self.opacitycard.configItem.valueChanged.connect(self.setOpacity)
        self.backgroundcard.button.clicked.connect(self.choose_background)
        self.touxiangcard.button.clicked.connect(self.choose_touxiang)
        # self.ptFolderCard.folders.append(self.ptpath)
        # self.musicInThisPCGroup.addSettingCard(self.ptFolderCard)
        # self.musicInThisPCGroup.addSettingCard(self.SwitchSettingCard)
        self.musicInThisPCGroup.addSettingCard(self.touxiangcard)
        self.musicInThisPCGroup.addSettingCard(self.ptcard)
        self.musicInThisPCGroup.addSettingCard(self.savecard)
        self.musicInThisPCGroup.addSettingCard(self.backgroundcard)
        self.musicInThisPCGroup.addSettingCard(self.opacitycard)
        # self.musicInThisPCGroup.addSettingCard(self.themeColorCard)
        self.expandLayout.setSpacing(28)
        self.expandLayout.setContentsMargins(36, 30, 36, 0)
        self.expandLayout.addWidget(self.musicInThisPCGroup)
        # self.expandLayout.addWidget(self.background_label)

    def addtab(self):
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.fileButton = PrimaryPushButton(self.tab)
        self.fileButton.setGeometry(QtCore.QRect(20, 100, 141, 41))
        self.fileButton.setStyleSheet("\n"
                                      "        QPushButton:hover{color:white;\n"
                                      "                    border:2px solid #F3F3F5;\n"
                                      "                    border-radius:35px;\n"
                                      "                    background:darkGray;}")
        self.fileButton.setObjectName("fileButton")
        self.checkBox_2 = RadioButton(self.tab)
        self.checkBox_2.setGeometry(QtCore.QRect(20, 180, 181, 51))
        self.checkBox_2.setObjectName("checkBox_2")
        self.layoutWidget = QtWidgets.QWidget(self.tab)
        self.layoutWidget.setGeometry(QtCore.QRect(220, 10, 1031, 750))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(10, 30, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_3 = HeaderCardWidget(self.layoutWidget)
        self.groupBox_3.setTitle("原图")
        self.groupBox_3.setObjectName("groupBox_3")
        self.raw = QtWidgets.QLabel(self.groupBox_3)
        self.raw.setGeometry(QtCore.QRect(10, 30, 491, 301))
        self.raw.setText("")
        self.raw.setObjectName("raw")
        self.gridLayout.addWidget(self.groupBox_3, 0, 0, 1, 1)
        self.groupBox_5 = CardWidget(self.layoutWidget)
        # self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_2.setGeometry(QtCore.QRect(180, 480, 75, 24))
        self.pushButton_2.setObjectName("pushButton_2")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_5)
        self.comboBox_2.setGeometry(QtCore.QRect(400, 480, 67, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_3.setGeometry(QtCore.QRect(290, 480, 75, 24))
        self.pushButton_3.setObjectName("pushButton_3")
        self.jiance1 = PushButton(self.groupBox_5)
        self.jiance1.setGeometry(QtCore.QRect(110, 310, 91, 31))
        # self.jiance1.setStyleSheet("QPushButton:hover{color:white;\n"
        #                            "                    border:2px solid #F3F3F5;\n"
        #                            "                    border-radius:35px;\n"
        #                            "                    background:darkGray;}")
        self.jiance1.setObjectName("jiance1")
        self.comboBox_3 = ComboBox(self.groupBox_5)
        self.comboBox_3.setGeometry(QtCore.QRect(330, 310, 121, 31))
        self.comboBox_3.setObjectName("comboBox_3")
        self.baocun1=PushButton(self.groupBox_5)
        self.baocun1.setIcon(FIF.SAVE)
        self.baocun1.setText('保存')
        # self.baocun1 = PushButton(FIF.SAVE,'保存')
        # self.groupBox_5.add
        self.baocun1.setGeometry(QtCore.QRect(220, 310, 101, 31))
        self.baocun1.setObjectName("baocun1")
        self.img1 = QtWidgets.QLabel(self.groupBox_5)
        self.img1.setGeometry(QtCore.QRect(10, 20, 501, 281))
        self.img1.setText("")
        self.img1.setObjectName("img1")
        self.gridLayout.addWidget(self.groupBox_5, 0, 1, 1, 1)
        self.groupBox_7 = CardWidget(self.layoutWidget)
        # self.groupBox_7.setTitle("")
        self.groupBox_7.setObjectName("groupBox_7")
        self.jiance2 = PushButton(self.groupBox_7)
        self.jiance2.setGeometry(QtCore.QRect(120, 300, 91, 31))
        # self.jiance2.setStyleSheet("QPushButton:hover{color:white;\n"
        #                            "                    border:2px solid #F3F3F5;\n"
        #                            "                    border-radius:35px;\n"
        #                            "                    background:darkGray;}")
        self.jiance2.setObjectName("jiance2")
        self.comboBox_4 = ComboBox(self.groupBox_7)
        self.comboBox_4.setGeometry(QtCore.QRect(340, 300, 121, 31))
        self.comboBox_4.setObjectName("comboBox_4")
        self.baocun2 = PushButton(self.groupBox_7)
        self.baocun2.setGeometry(QtCore.QRect(230, 300, 101, 31))
        self.baocun2.setObjectName("baocun2")
        self.baocun2.setIcon(FIF.SAVE)
        self.baocun2.setText('保存')
        self.img2 = QtWidgets.QLabel(self.groupBox_7)
        self.img2.setGeometry(QtCore.QRect(10, 20, 491, 271))
        self.img2.setText("")
        self.img2.setObjectName("img2")
        self.gridLayout.addWidget(self.groupBox_7, 1, 0, 1, 1)
        self.groupBox_11 = CardWidget(self.layoutWidget)
        # self.groupBox_11.setTitle("")
        self.groupBox_11.setObjectName("groupBox_11")
        self.jiance3 = PushButton(self.groupBox_11)
        self.jiance3.setGeometry(QtCore.QRect(110, 300, 91, 31))
        # self.jiance3.setStyleSheet("QPushButton:hover{color:white;\n"
        #                            "                    border:2px solid #F3F3F5;\n"
        #                            "                    border-radius:35px;\n"
        #                            "                    background:darkGray;}")
        self.jiance3.setObjectName("jiance3")
        self.comboBox_5 = ComboBox(self.groupBox_11)
        self.comboBox_5.setGeometry(QtCore.QRect(330, 300, 121, 31))
        self.comboBox_5.setObjectName("comboBox_5")
        self.baocun3 = PushButton(self.groupBox_11)
        self.baocun3.setGeometry(QtCore.QRect(220, 300, 101, 31))
        self.baocun3.setObjectName("baocun3")
        self.baocun3.setIcon(FIF.SAVE)
        self.baocun3.setText('保存')
        self.img3 = QtWidgets.QLabel(self.groupBox_11)
        self.img3.setGeometry(QtCore.QRect(10, 20, 491, 271))
        self.img3.setText("")
        self.img3.setObjectName("img3")
        self.gridLayout.addWidget(self.groupBox_11, 1, 1, 1, 1)

    def addtab2(self):
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.layoutWidget1 = QtWidgets.QWidget(self.tab_2)
        self.layoutWidget1.setGeometry(QtCore.QRect(220, 690, 1046, 42))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_12.setContentsMargins(11, 0, 11, 0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.runButton = PrimaryToolButton(self.layoutWidget1)
        self.runButton.setMinimumSize(QtCore.QSize(40, 40))
        self.runButton.setIcon(FluentIcon.PLAY)
        # self.runButton.setStyleSheet("QPushButton {\n"
        #                              "border-style: solid;\n"
        #                              "border-width: 0px;\n"
        #                              "border-radius: 0px;\n"
        #                              "background-color: rgba(0, 0, 0, 100);\n"
        #                              "}\n"
        #                              "QPushButton::focus{outline: none;}\n"
        #                              "QPushButton::hover {\n"
        #                              "border-style: solid;\n"
        #                              "border-width: 0px;\n"
        #                              "border-radius: 0px;\n"
        #                              "background-color: rgba(223, 223, 223, 150);}")
        # self.runButton.setText("")
        # icon4 = QtGui.QIcon()
        # icon4.addPixmap(QtGui.QPixmap(":/img/icon/运行.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        # icon4.addPixmap(QtGui.QPixmap(":/img/icon/暂停.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # icon4.addPixmap(QtGui.QPixmap(":/img/icon/运行.png"), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        # icon4.addPixmap(QtGui.QPixmap(":/img/icon/暂停.png"), QtGui.QIcon.Disabled, QtGui.QIcon.On)
        # icon4.addPixmap(QtGui.QPixmap(":/img/icon/运行.png"), QtGui.QIcon.Active, QtGui.QIcon.Off)
        # icon4.addPixmap(QtGui.QPixmap(":/img/icon/暂停.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        # icon4.addPixmap(QtGui.QPixmap(":/img/icon/运行.png"), QtGui.QIcon.Selected, QtGui.QIcon.Off)
        # icon4.addPixmap(QtGui.QPixmap(":/img/icon/暂停.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        # self.runButton.setIcon(icon4)
        # self.runButton.setIconSize(QtCore.QSize(30, 30))
        self.runButton.setCheckable(True)
        self.runButton.setObjectName("runButton")
        self.horizontalLayout_12.addWidget(self.runButton)
        self.progressBar = ProgressBar(self.layoutWidget1)
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 5))
        self.progressBar.setStyleSheet(
            "QProgressBar{ color: rgb(255, 255, 255); font:12pt; border-radius:2px; text-align:center; border:none; background-color: rgba(215, 215, 215,100);} \n"
            "QProgressBar:chunk{ border-radius:0px; background: rgba(55, 55, 55, 200);}")
        self.progressBar.setMaximum(1000)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_12.addWidget(self.progressBar)
        self.stopButton = PrimaryToolButton(self.layoutWidget1)
        self.stopButton.setMinimumSize(QtCore.QSize(40, 40))
        self.stopButton.setIcon(FluentIcon.POWER_BUTTON)
        # self.stopButton.setStyleSheet("QPushButton {\n"
        #                               "border-style: solid;\n"
        #                               "border-width: 0px;\n"
        #                               "border-radius: 0px;\n"
        #                               "background-color: rgba(0, 0, 0, 100);\n"
        #                               "}\n"
        #                               "QPushButton::focus{outline: none;}\n"
        #                               "QPushButton::hover {\n"
        #                               "border-style: solid;\n"
        #                               "border-width: 0px;\n"
        #                               "border-radius: 0px;\n"
        #                               "background-color: rgba(223, 223, 223, 150);}")
        # self.stopButton.setText("")
        # icon5 = QtGui.QIcon()
        # icon5.addPixmap(QtGui.QPixmap(":/img/icon/终止.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        # self.stopButton.setIcon(icon5)
        # self.stopButton.setIconSize(QtCore.QSize(30, 30))
        # self.stopButton.setObjectName("stopButton")
        self.horizontalLayout_12.addWidget(self.stopButton)
        self.layoutWidget_2 = QtWidgets.QWidget(self.tab_2)
        self.layoutWidget_2.setGeometry(QtCore.QRect(220, 10, 1031, 650))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget_2)
        self.gridLayout_2.setContentsMargins(10, 30, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox_14 = CardWidget(self.layoutWidget_2)
        # self.groupBox_14.setTitle("")
        self.groupBox_14.setObjectName("groupBox_14")
        self.label_2 = QtWidgets.QLabel(self.groupBox_14)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 511, 671))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.groupBox_14, 0, 1, 1, 1)
        self.groupBox_9 = HeaderCardWidget(self.layoutWidget_2)
        # self.groupBox_9.setTitle("")
        self.groupBox_9.setObjectName("groupBox_9")
        self.label = QtWidgets.QLabel(self.groupBox_9)
        self.label.setGeometry(QtCore.QRect(0, 0, 511, 661))
        self.label.setText("")
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.groupBox_9, 0, 0, 1, 1)
        self.pushButton_13 = PrimaryPushButton(self.tab_2)
        self.pushButton_13.setGeometry(QtCore.QRect(20, 100, 141, 41))
        self.pushButton_13.setStyleSheet("\n"
                                      "        QPushButton:hover{color:white;\n"
                                      "                    border:2px solid #F3F3F5;\n"
                                      "                    border-radius:35px;\n"
                                      "                    background:darkGray;}")
        self.pushButton_13.setObjectName("pushButton_13")
        self.comboBox_6 = ComboBox(self.tab_2)
        self.comboBox_6.setGeometry(QtCore.QRect(100, 180, 121, 31))
        self.comboBox_6.setObjectName("comboBox_6")
        self.label_5 = BodyLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(30, 180, 61, 31))
        self.label_5.setObjectName("label_5")

    def addtab3(self):
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.pushButton = TogglePushButton(self.tab_3)
        self.pushButton.setGeometry(QtCore.QRect(10, 190, 181, 61))
        # self.pushButton.setStyleSheet("       QPushButton:hover{color:white;\n"
        #                               "                    border:2px solid #F3F3F5;\n"
        #                               "                    border-radius:35px;\n"
        #                               "                    background:darkGray;}")
        self.pushButton.setObjectName("pushButton")
        # self.checkBox = RadioButton(self.tab_3)
        # self.checkBox.setGeometry(QtCore.QRect(30, 530, 79, 20))
        # self.checkBox.setObjectName("checkBox")
        # self.pushButton_11 = QtWidgets.QPushButton(self.tab_3)
        # self.pushButton_11.setGeometry(QtCore.QRect(120, 460, 75, 24))
        # self.pushButton_11.setStyleSheet("       QPushButton:hover{color:white;\n"
        #                                  "                    border:2px solid #F3F3F5;\n"
        #                                  "                    border-radius:35px;\n"
        #                                  "                    background:darkGray;}")
        # self.pushButton_11.setObjectName("pushButton_11")
        # self.pushButton_12 = QtWidgets.QPushButton(self.tab_3)
        # self.pushButton_12.setGeometry(QtCore.QRect(120, 550, 75, 24))
        # self.pushButton_12.setStyleSheet("       QPushButton:hover{color:white;\n"
        #                                  "                    border:2px solid #F3F3F5;\n"
        #                                  "                    border-radius:35px;\n"
        #                                  "                    background:darkGray;}")
        # self.pushButton_12.setObjectName("pushButton_12")
        self.layoutWidget_3 = QtWidgets.QWidget(self.tab_3)
        self.layoutWidget_3.setGeometry(QtCore.QRect(220, 10, 1031, 750))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.layoutWidget_3)
        self.gridLayout_4.setContentsMargins(10, 30, 20, 40)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.groupBox_16 = CardWidget(self.layoutWidget_3)
        # self.groupBox_16.setTitle("")
        self.groupBox_16.setObjectName("groupBox_16")
        self.camera1 = QtWidgets.QLabel(self.groupBox_16)
        self.camera1.setGeometry(QtCore.QRect(0, 10, 511, 750))
        self.camera1.setText("")
        self.camera1.setObjectName("camera1")
        self.gridLayout_4.addWidget(self.groupBox_16, 0, 1, 1, 1)
        self.groupBox_12 = HeaderCardWidget(self.layoutWidget_3)
        # self.groupBox_12.setTitle("")
        self.groupBox_12.setObjectName("groupBox_12")
        self.camera = QtWidgets.QLabel(self.groupBox_12)
        self.camera.setGeometry(QtCore.QRect(10, 10, 501, 750))
        self.camera.setText("")
        self.camera.setObjectName("camera")
        self.gridLayout_4.addWidget(self.groupBox_12, 0, 0, 1, 1)
        self.comboBox = ComboBox(self.tab_3)
        self.comboBox.setGeometry(QtCore.QRect(10, 350, 181, 51))
        self.comboBox.setObjectName("comboBox")
        self.lineEdit_2 = LineEdit(self.tab_3)
        self.lineEdit_2.setPlaceholderText("输入你的rtsp地址😁")
        self.lineEdit_2.setGeometry(QtCore.QRect(10, 70, 201, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton_8 = PushButton(self.tab_3)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 110, 81, 41))
        self.pushButton_8.setStyleSheet("       QPushButton:hover{color:white;\n"
                                        "                    border:2px solid #F3F3F5;\n"
                                        "                    border-radius:35px;\n"
                                        "                    background:darkGray;}")
        self.pushButton_8.setObjectName("pushButton_8")
        self.label_3 = BodyLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(10, 50, 54, 16))
        self.label_3.setObjectName("label_3")
        self.pushButton_14 = PushButton(self.tab_3)
        self.pushButton_14.setGeometry(QtCore.QRect(110, 110, 81, 41))
        self.pushButton_14.setStyleSheet("       QPushButton:hover{color:white;\n"
                                         "                    border:2px solid #F3F3F5;\n"
                                         "                    border-radius:35px;\n"
                                         "                    background:darkGray;}")
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_10 = TogglePushButton(self.tab_3)
        self.pushButton_10.setGeometry(QtCore.QRect(10, 270, 181, 61))
        # self.pushButton_10.setStyleSheet("       QPushButton:hover{color:white;\n"
        #                                  "                    border:2px solid #F3F3F5;\n"
        #                                  "                    border-radius:35px;\n"
        #                                  "                    background:darkGray;}")
        self.pushButton_10.setObjectName("pushButton_10")
        self.line = QtWidgets.QFrame(self.tab_3)
        self.line.setGeometry(QtCore.QRect(0, 160, 221, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.fileButton.setText(_translate("mainWindow", "选择图片"))
        self.checkBox_2.setText(_translate("mainWindow", "自动一键检测"))
        self.pushButton_2.setText(_translate("mainWindow", "检测"))
        self.pushButton_3.setText(_translate("mainWindow", "保存"))
        self.jiance1.setText(_translate("mainWindow", "检测1"))
        # self.baocun1.setText(_translate("mainWindow", "保存"))
        self.jiance2.setText(_translate("mainWindow", "检测2"))
        # self.baocun2.setText(_translate("mainWindow", "保存"))
        self.jiance3.setText(_translate("mainWindow", "检测3"))
        # self.baocun3.setText(_translate("mainWindow", "保存"))
        self.pushButton_13.setText(_translate("mainWindow", "选择视频"))
        self.label_5.setText(_translate("mainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">选择模型</span></p></body></html>"))
        self.pushButton.setText(_translate("mainWindow", "打开摄像头"))
        # self.checkBox.setText(_translate("mainWindow", "自动检测"))
        # self.pushButton_11.setText(_translate("mainWindow", "停止检测"))
        # self.pushButton_12.setText(_translate("mainWindow", "关闭摄像头"))
        # self.lineEdit_2.setText(_translate("mainWindow", "http://10.207.29.12:4747/video"))
        self.pushButton_8.setText(_translate("mainWindow", "打开"))
        self.label_3.setText(_translate("mainWindow", "rtsp地址："))
        self.pushButton_14.setText(_translate("mainWindow", "关闭"))
        self.pushButton_10.setText(_translate("mainWindow", "开始检测"))
        # setting
        # self.pushButton_16.setText(_translate("mainWindow", "Chose"))
        # self.pushButton_17.setText(_translate("mainWindow", "Chose"))
        # self.label_6.setText(_translate("mainWindow", "选择模型路径"))
        # self.label_8.setText(_translate("mainWindow", "当前模型路径："))
        # self.label_9.setText(_translate("mainWindow", "TextLabel"))
        # self.label_10.setText(_translate("mainWindow", "选择保存路径"))
        # self.label_11.setText(_translate("mainWindow", "TextLabel"))
        # self.label_12.setText(_translate("mainWindow", "当前保存路径："))

    def choose_background(self):
        open_fold = 'D:\Pictures\Saved Pictures'
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        if name:
            self.background_path=name
            self.backgroundcard.contentLabel.setText(name)
            self.setBackground()
            InfoBar.success(
                title='变变变！！',
                content="新的背景，你喜欢吗？",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                duration=2000,
                parent=self
            )

    def settouxiang(self):
        self.setWindowIcon(QIcon(self.touxiang_path))
        self.navigationInterface.panel.items['avatar'].widget.setWindowIcon(QIcon(self.touxiang_path))

    def choose_touxiang(self):
        open_fold = 'D:\Pictures\Saved Pictures'
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        if name:
            self.touxiang_path=name
            self.settouxiang()
            self.touxiangcard.contentLabel.setText(name)
            InfoBar.success(
                title='变变变！！',
                content="你好！重新认识一下吧😉",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                duration=2000,
                parent=self
            )

    def setOpacity(self,opacity):
        self.setWindowOpacity((opacity+1)/100)

    def initNavigation(self):

        self.addSubInterface(self.tab, FIF.PHOTO, '图片')
        self.addSubInterface(self.tab_2, FIF.VIDEO, '视频')
        self.addSubInterface(self.tab_3, FIF.CAMERA, '摄像机')

        self.navigationInterface.addWidget(
            routeKey='avatar',
            widget=NavigationAvatarWidget('Undecimber', self.touxiang_path),
            onClick=self.showMessageBox,
            position=NavigationItemPosition.BOTTOM,
        )
        self.addSubInterface(
            self.tab_4, FIF.SETTING, self.tr('Settings'), NavigationItemPosition.BOTTOM)

        self.navigationInterface.setExpandWidth(280)

        self.navigationInterface.addItem(
            "settingInterface", FluentIcon.CLEAR_SELECTION, "清除ALL", onClick=lambda: self.clear_label(), selectable=False)

    def clear_label(self):
        self.label.setPixmap(QPixmap(""))
        self.label_2.setPixmap(QPixmap(""))
        self.camera.setPixmap(QPixmap(""))
        self.camera1.setPixmap(QPixmap(""))
        self.raw.setPixmap(QPixmap(''))
        self.img1.setPixmap(QPixmap(''))
        self.img2.setPixmap(QPixmap(''))
        self.img3.setPixmap(QPixmap(''))
        InfoBar.success(
            title='一键清除！',
            content="这次罚你，再也见不到我🤒",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM_RIGHT,
            duration=3000,
            parent=self
        )
    def initWindow(self):
        # self.resize(1350, 780)
        self.setWindowIcon(QIcon(self.touxiang_path))
        self.setWindowTitle('基于YOLOv5的无人机目标检测')

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

    def showMessageBox(self):
        w = MessageBox(
            '啥呀这是？😕',
            'NUDT信息工程课程设计——基于YOLOv5的无人机目标检测',
            # '个人开发不易，如果这个项目帮助到了您，可以考虑请作者喝一瓶快乐水🥤。您的支持就是作者开发和维护项目的动力🚀',
            self
        )
        w.yesButton.setText('我也要上NUDT😍！')
        w.cancelButton.setText('做得真棒🥰')

        if w.exec():
            QDesktopServices.openUrl(QUrl("https://www.nudt.edu.cn/"))

    def tog_camera(self,checked=False):
        if checked:
            self.chose_cam()
        else:
            self.close_cam()

    def tog_det(self,checked=False):
        if checked:
            InfoBar.success(
                title='开始检测！',
                content="看看能检测到什么吧！",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                duration=1000,
                parent=self
            )
            self.camera_detect()
        else:
            self.camera_stop()

    def tryallmodel(self):
        if self.checkBox_2.isChecked():
            self.try_all_model=True
        else:
            self.try_all_model=False

    def saveimage(self,label):
        try:
            os.makedirs(self.savepath, exist_ok=True)
            name=self.det_thread.weights.split('\\')[-1][:-3]+self.det_thread.source.split('/')[-1]
            image_path=os.path.join(self.savepath,name)
            a=cv2.imwrite(image_path, self.det_thread.result)
            if a:
                InfoBar.success(
                    title='保存成功',
                    content=f"文件保存在{image_path}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.BOTTOM_RIGHT,
                    duration=1000,
                    parent=self
                )
        except Exception as e:
            InfoBar.error(
                title='保存失败',
                content='不知道咋回事捏，再试试吧',
                orient=Qt.Vertical,  # 内容太长时可使用垂直布局
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                duration=2000,
                parent=self
            )

    def open_PTfile(self):
        directory = QFileDialog.getExistingDirectory(None, "选取文件夹", "./")  # 起始路径
        if directory:
            self.ptcard.contentLabel.setText(directory)
            # self.label_9.setText(directory)
            self.ptpath=directory

    def chose_savepath(self):
        directory = QFileDialog.getExistingDirectory(None, "选取文件夹", "./")  # 起始路径
        if directory:
            self.savecard.contentLabel.setText(directory)
            # self.label_11.setText(directory)
            self.savepath=directory


    def camera_stop(self):
        self.det_thread.camera_detect_flag=False
    def camera_detect(self):
        self.det_thread.camera_detect_flag=True

    def close_cam(self):
        self.camera_open = 1
        self.det_thread.jump_out=True

    def search_pt(self):

        self.comboBox_list=[self.comboBox,self.comboBox_3, self.comboBox_4, self.comboBox_5,self.comboBox_6]
        if self.click_list:
            botton=self.click_list.pop(0)
            botton.click()

        os.makedirs(self.ptpath, exist_ok=True)
        pt_list = os.listdir(self.ptpath)

        pt_list = [file for file in pt_list if file.endswith('.pt')]
        if len(pt_list)==0:
            InfoBar.warning(
                title='未检测到模型',
                content="里面空空如也哦！！",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM,
                duration=2500,  # 永不消失
                parent=self
            )
            return
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            for i in [self.comboBox,self.comboBox_3, self.comboBox_4, self.comboBox_5,self.comboBox_6]:
                i.clear()
                i.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = self.savepath
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.load_rtsp(self.lineEdit_2.text())

    def load_rtsp(self, ip):
        try:
            self.stop()

            self.det_thread.source = ip
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.det_thread.source = ip
            self.run_or_continue_rtsp(self.camera, self.comboBox.currentText())
        except Exception as e:
            InfoBar.error(
                title='怎么回事？出错了😥',
                content="rtsp地址输错了？重新试试吧！",
                orient=Qt.Vertical,  # 内容太长时可使用垂直布局
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                duration=-1,
                parent=window
            )

    def run_or_continue_rtsp(self,label,model):
        self.det_thread.jump_out = False
        self.det_thread.is_continue = True
        self.det_thread.label=label
        self.det_thread.weights=os.path.join(self.ptpath,model)
        if not self.det_thread.isRunning():
            self.det_thread.start()
        source = os.path.basename(self.det_thread.source)
        source = 'camera' if source.isnumeric() else source
        if source=='video' and (self.camera_open==1):
            InfoBar.success(
                title='正在通过rtsp检测！',
                content="可能有点慢哦！耐心点嘛🤗",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                duration=1000,
                parent=self
            )
            self.camera_open = 0
        else:
            InfoBar.success(
                title=f'开始检测图片',
                content=f'用{model}能检测到什么呢？🫣',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                duration=2000,
                parent=self
            )

    def chose_cam(self):
        try:
            self.stop()
            self.det_thread.label=[self.camera,self.camera1]
            self.det_thread.source='0'
            self.run_or_continue(self.camera,self.comboBox.currentText())
        except Exception as e:
            InfoBar.error(
                title='怎么回事？出错了😥',
                content="摄像头忘关了？重新试试吧！",
                orient=Qt.Vertical,  # 内容太长时可使用垂直布局
                isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT,
                duration=-1,
                parent=window
            )

    def statistic_msg(self, msg):
        pass
        # self.statistic_label.setText(msg)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = os.path.join(self.ptpath,self.model_type)
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):

        open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                              "*.jpg *.png)")
        if name.lower().endswith(('mp4')):
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            self.stop()
        elif name:
            dataset = LoadImages(name)
            dataset = iter(dataset)
            path, img, im0s, self.vid_cap = next(dataset)
            self.show_image(im0s, self.raw)
            self.det_thread.source = name
            self.stop()
            if self.try_all_model:
                for k in [self.jiance1,self.jiance2,self.jiance3]:
                    self.click_list.append(k)


    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self,label,model):
        self.det_thread.jump_out = False
        # if self.runButton.isChecked():
        if 1==1:
            # self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            self.det_thread.label=label
            self.det_thread.weights=os.path.join(self.ptpath,model)
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
            if source=='camera' and (self.camera_open==1):
                InfoBar.success(
                    title=f'正在打开摄像头...',
                    content=f'苏醒吧，摄像头！',
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.BOTTOM_RIGHT,
                    duration=2000,
                    parent=self
                )
                self.camera_open=0
            else:
                InfoBar.success(
                    title=f'开始检测图片',
                    content=f'用{model}能检测到什么呢？🫣',
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.BOTTOM_RIGHT,
                    duration=2000,
                    parent=self
                )

    def run_or_continue1(self,label,model):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.runButton.setIcon(FluentIcon.PAUSE)
            self.det_thread.is_continue = True
            self.det_thread.label=label
            self.det_thread.weights=os.path.join(self.ptpath,model)
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.runButton.setIcon(FluentIcon.PLAY)
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print('{:*^60}'.format('直接打印出e, 输出错误具体原因'))
            print(e)
            print('{:*^60}'.format('使用repr打印出e, 带有错误类型'))
            print(repr(e))
            print('{:*^60}'.format('使用traceback的format_exc可以输出错误具体位置'))
            exstr = traceback.format_exc()
            print(exstr)

    def show_statistic(self, statistic_dic):
        try:
            self.listWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.listWidget.addItems(results)

        except Exception as e:
            print('{:*^60}'.format('直接打印出e, 输出错误具体原因'))
            print(e)
            print('{:*^60}'.format('使用repr打印出e, 带有错误类型'))
            print(repr(e))
            print('{:*^60}'.format('使用traceback的format_exc可以输出错误具体位置'))
            exstr = traceback.format_exc()
            print(exstr)

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        sys.exit(0)

if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # setTheme(Theme.DARK)

    app = QApplication(sys.argv)

    # install translator
    translator = FluentTranslator()
    app.installTranslator(translator)

    w = Window()
    w.show()
    app.exec_()
