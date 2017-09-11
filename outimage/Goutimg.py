import wx
import sys,os
import cv2
import numpy as np

# 変数の宣言
filepath = ""
filepath2 = ""
text1 = ""


class MainFrame(wx.Frame):

    # イベント
    def click_button_1(self, event):
        global filepath
        global text1
        # ファイル選択ダイアログを作成
        dialog = wx.FileDialog(None, u'比較ファイルを選択してください')
        # ファイル選択ダイアログを表示
        dialog.ShowModal()
        # 選択したファイルパスを取得する
        filepath = dialog.GetPath()
        text1.AppendText(filepath + "\n")

    def click_button_2(self, event):
        global filepath2
        global text1
        # ファイル選択ダイアログを作成
        dialog = wx.FileDialog(None, u'比較ファイルを選択してください')
        # ファイル選択ダイアログを表示
        dialog.ShowModal()
        # 選択したファイルパスを取得する
        filepath2 = dialog.GetPath()
        text1.AppendText(filepath2+ "\n")        

    def click_button_3(self, event):
        global filepath
        global filepath2
        IMG_SIZE = (200, 200)
        img = cv2.imread(filepath)
        img2 = cv2.imread(filepath2)
        img = cv2.resize(img, IMG_SIZE)
        img2 = cv2.resize(img2, IMG_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 特徴量記述
        detector = cv2.AKAZE_create()
        kp1, des1 = detector.detectAndCompute(gray, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)

        # 比較器作成
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # 画像への特徴点の書き込み
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)

        # 出力画像作成 表示
        h1, w1, c1 = img.shape[:3]
        h2, w2, c2 = img2.shape[:3]
        height = max([h1,h2])
        width = w1 + w2
        out = np.zeros((height, width, 3), np.uint8)

        cv2.drawMatches(img, kp1, img2, kp2, matches[:50],out, flags=0)
        cv2.imshow("name", out)
        cv2.waitKey(0)


    def __init__(self):
         wx.Frame.__init__(self, None, wx.ID_ANY, "Main")
         self.InitializeComponents()

    def InitializeComponents(self):
        global text1
        global button_1
        self.CreateStatusBar()
        # ボタンの作成
        panel = wx.Panel(self, wx.ID_ANY)
        panel.SetBackgroundColour("#AFAFAF")
        button_1 = wx.Button(panel, wx.ID_ANY, u"比較ファイル1")
        button_2 = wx.Button(panel, wx.ID_ANY, u"比較ファイル2")
        button_3 = wx.Button(panel, wx.ID_ANY, u"実行")
        # panel2 = wx.Panel(frame, wx.ID_ANY)
        # panel2.SetBackgroundColour("#000000")
        text1 = wx.TextCtrl(panel, wx.ID_ANY, style=wx.TE_MULTILINE)

        # イベントの設定
        button_1.Bind(wx.EVT_BUTTON, self.click_button_1)
        button_2.Bind(wx.EVT_BUTTON, self.click_button_2)
        button_3.Bind(wx.EVT_BUTTON, self.click_button_3)

        # ボタンレイアウト
        layout = wx.GridBagSizer()
        layout.Add(button_1, (0,0), (1,1), flag=wx.EXPAND)
        layout.Add(button_2, (0,1), (1,1), flag=wx.EXPAND)
        layout.Add(button_3, (0,2), (1,1), flag=wx.EXPAND)
        layout.Add(text1, (1,0), (3,3), flag=wx.EXPAND)
        layout.AddGrowableRow(0)
        layout.AddGrowableRow(1)
        layout.AddGrowableRow(2)
        layout.AddGrowableRow(3)
        layout.AddGrowableCol(0)
        layout.AddGrowableCol(1)
        layout.AddGrowableCol(2)
        panel.SetSizer(layout)

if __name__ == '__main__':
    app = wx.App()
    MainFrame().Show(True)
    app.MainLoop()
