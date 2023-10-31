# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Sim_Inputs.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(392, 705)
        #Variable Input Widgets
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.InitialAoA = QtWidgets.QLineEdit(self.centralwidget)
        self.InitialAoA.setGeometry(QtCore.QRect(120, 140, 113, 21))
        self.InitialAoA.setObjectName("InitialAoA")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 140, 91, 20))
        self.label.setObjectName("label")
        self.Button_1 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_1.setGeometry(QtCore.QRect(260, 610, 111, 32))
        self.Button_1.setObjectName("Button_1")
        self.ZeroDegCd = QtWidgets.QLineEdit(self.centralwidget)
        self.ZeroDegCd.setGeometry(QtCore.QRect(120, 170, 113, 21))
        self.ZeroDegCd.setObjectName("ZeroDegCd")
        self.Chord = QtWidgets.QLineEdit(self.centralwidget)
        self.Chord.setGeometry(QtCore.QRect(120, 200, 113, 21))
        self.Chord.setObjectName("Chord")
        self.Span = QtWidgets.QLineEdit(self.centralwidget)
        self.Span.setGeometry(QtCore.QRect(120, 230, 113, 21))
        self.Span.setObjectName("Span")
        self.Mass = QtWidgets.QLineEdit(self.centralwidget)
        self.Mass.setGeometry(QtCore.QRect(120, 320, 113, 21))
        self.Mass.setObjectName("Mass")
        self.Wheelbase = QtWidgets.QLineEdit(self.centralwidget)
        self.Wheelbase.setGeometry(QtCore.QRect(120, 350, 113, 21))
        self.Wheelbase.setObjectName("Wheelbase")
        self.TireRadius = QtWidgets.QLineEdit(self.centralwidget)
        self.TireRadius.setGeometry(QtCore.QRect(120, 380, 113, 21))
        self.TireRadius.setObjectName("TireRadius")
        self.MaxBHP = QtWidgets.QLineEdit(self.centralwidget)
        self.MaxBHP.setGeometry(QtCore.QRect(120, 410, 113, 21))
        self.MaxBHP.setObjectName("MaxBHP")
        self.Efficiency = QtWidgets.QLineEdit(self.centralwidget)
        self.Efficiency.setGeometry(QtCore.QRect(120, 440, 113, 21))
        self.Efficiency.setObjectName("Efficiency")
        self.StaticFriction = QtWidgets.QLineEdit(self.centralwidget)
        self.StaticFriction.setGeometry(QtCore.QRect(120, 470, 113, 21))
        self.StaticFriction.setObjectName("StaticFriction")
        self.CarCd = QtWidgets.QLineEdit(self.centralwidget)
        self.CarCd.setGeometry(QtCore.QRect(120, 500, 113, 21))
        self.CarCd.setObjectName("CarCd")
        self.CarArea = QtWidgets.QLineEdit(self.centralwidget)
        self.CarArea.setGeometry(QtCore.QRect(120, 530, 113, 21))
        self.CarArea.setObjectName("CarArea")
        self.RollResis = QtWidgets.QLineEdit(self.centralwidget)
        self.RollResis.setGeometry(QtCore.QRect(120, 560, 113, 21))
        self.RollResis.setObjectName("RollResis")
        #Text Widgets (Cosmetic)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(17, 170, 91, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 200, 91, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 230, 91, 20))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 320, 91, 20))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 350, 91, 20))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(20, 380, 91, 20))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(20, 410, 91, 20))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(20, 440, 91, 20))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(20, 470, 91, 20))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(20, 500, 91, 20))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(20, 530, 91, 20))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(20, 560, 91, 20))
        self.label_13.setObjectName("label_13")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(20, 160, 351, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(20, 190, 351, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(20, 220, 351, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(20, 310, 351, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(20, 340, 351, 16))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setGeometry(QtCore.QRect(20, 370, 351, 16))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_7 = QtWidgets.QFrame(self.centralwidget)
        self.line_7.setGeometry(QtCore.QRect(20, 400, 351, 16))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.line_8 = QtWidgets.QFrame(self.centralwidget)
        self.line_8.setGeometry(QtCore.QRect(20, 430, 351, 16))
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.line_9 = QtWidgets.QFrame(self.centralwidget)
        self.line_9.setGeometry(QtCore.QRect(20, 460, 351, 16))
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.line_10 = QtWidgets.QFrame(self.centralwidget)
        self.line_10.setGeometry(QtCore.QRect(20, 490, 351, 16))
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.line_11 = QtWidgets.QFrame(self.centralwidget)
        self.line_11.setGeometry(QtCore.QRect(20, 520, 351, 16))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.line_12 = QtWidgets.QFrame(self.centralwidget)
        self.line_12.setGeometry(QtCore.QRect(20, 550, 351, 16))
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.Title = QtWidgets.QLabel(self.centralwidget)
        self.Title.setGeometry(QtCore.QRect(20, 0, 221, 41))
        #Header Text Lines (Cosmetic)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(True)
        self.Title.setFont(font)
        self.Title.setObjectName("Title")
        self.Subtitle = QtWidgets.QLabel(self.centralwidget)
        self.Subtitle.setGeometry(QtCore.QRect(20, 30, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Subtitle.setFont(font)
        self.Subtitle.setObjectName("Subtitle")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(260, 110, 101, 16))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(20, 110, 211, 16))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(20, 290, 211, 16))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.line_13 = QtWidgets.QFrame(self.centralwidget)
        self.line_13.setGeometry(QtCore.QRect(20, 250, 351, 16))
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.line_14 = QtWidgets.QFrame(self.centralwidget)
        self.line_14.setGeometry(QtCore.QRect(20, 130, 351, 16))
        self.line_14.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.line_15 = QtWidgets.QFrame(self.centralwidget)
        self.line_15.setGeometry(QtCore.QRect(20, 580, 351, 16))
        self.line_15.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(290, 140, 71, 20))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(290, 170, 71, 20))
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(290, 200, 71, 20))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(290, 230, 71, 20))
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(290, 320, 71, 20))
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(290, 350, 71, 20))
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(290, 380, 71, 20))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(290, 410, 71, 20))
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(290, 440, 71, 20))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(290, 470, 71, 20))
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.centralwidget)
        self.label_27.setGeometry(QtCore.QRect(290, 500, 71, 20))
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.centralwidget)
        self.label_28.setGeometry(QtCore.QRect(290, 530, 71, 20))
        self.label_28.setObjectName("label_28")
        self.label_29 = QtWidgets.QLabel(self.centralwidget)
        self.label_29.setGeometry(QtCore.QRect(290, 560, 71, 20))
        self.label_29.setObjectName("label_29")
        self.label_30 = QtWidgets.QLabel(self.centralwidget)
        self.label_30.setGeometry(QtCore.QRect(260, 290, 101, 16))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 392, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        #Defines Button Click Function, And Closes Upon Clicking
        self.Button_1.clicked.connect(self.WriteVals) 
        self.Button_1.clicked.connect(MainWindow.close) 

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    #Text Lines Defining Parameters And Initial Values (Cosmetic)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Initial AoA"))
        self.Button_1.setText(_translate("MainWindow", "Submit Inputs"))
        self.label_2.setText(_translate("MainWindow", "Cd 0 deg"))
        self.label_3.setText(_translate("MainWindow", "Chord (m)"))
        self.label_4.setText(_translate("MainWindow", "Span (m)"))
        self.label_5.setText(_translate("MainWindow", "Mass (kg)"))
        self.label_6.setText(_translate("MainWindow", "Wheelbase (m)"))
        self.label_7.setText(_translate("MainWindow", "Tire Radius (m)"))
        self.label_8.setText(_translate("MainWindow", "Max. Brake HP"))
        self.label_9.setText(_translate("MainWindow", "Pow. Efficiency"))
        self.label_10.setText(_translate("MainWindow", "Stat. Friction"))
        self.label_11.setText(_translate("MainWindow", "Cd "))
        self.label_12.setText(_translate("MainWindow", "Area (m^2)"))
        self.label_13.setText(_translate("MainWindow", "Rolling Resist."))
        self.Title.setText(_translate("MainWindow", "Enter variable values below:"))
        self.Subtitle.setText(_translate("MainWindow", "(Or leave blank to use defaults)"))
        self.label_14.setText(_translate("MainWindow", "Default Values"))
        self.label_15.setText(_translate("MainWindow", "Wing Parameters"))
        self.label_16.setText(_translate("MainWindow", "Vehicle Parameters"))
        self.label_17.setText(_translate("MainWindow", "10"))
        self.label_18.setText(_translate("MainWindow", "0.008"))
        self.label_19.setText(_translate("MainWindow", "0.30"))
        self.label_20.setText(_translate("MainWindow", "1.00"))
        self.label_21.setText(_translate("MainWindow", "798"))
        self.label_22.setText(_translate("MainWindow", "3.6"))
        self.label_23.setText(_translate("MainWindow", "0.23"))
        self.label_24.setText(_translate("MainWindow", "850"))
        self.label_25.setText(_translate("MainWindow", "0.98"))
        self.label_26.setText(_translate("MainWindow", "1.7"))
        self.label_27.setText(_translate("MainWindow", "0.85"))
        self.label_28.setText(_translate("MainWindow", "1.30"))
        self.label_29.setText(_translate("MainWindow", "0.012"))
        self.label_30.setText(_translate("MainWindow", "Default Values"))
     
    #Writes Values Of Each Input Line Into .txt File
    def WriteVals(self): 
        values=[]
        if self.InitialAoA.text() == '': 
            L1=10
            values.append(L1)
        else:
            L1=float(self.InitialAoA.text())
            values.append(L1)
            
        if self.ZeroDegCd.text() == '': 
            L2=0.008
            values.append(L2)
        else:
            L2=float(self.ZeroDegCd.text())
            values.append(L2)
            
        if self.Chord.text() == '': 
            L3=0.3
            values.append(L3)
        else:
            L3=float(self.Chord.text())
            values.append(L3)
            
        if self.Span.text() == '': 
            L4=1
            values.append(L4)
        else:
            L4=float(self.Span.text())
            values.append(L4)
            
        if self.Mass.text() == '': 
            L5=798
            values.append(L5)
        else:
            L5=float(self.Mass.text())
            values.append(L5)
            
        if self.Wheelbase.text() == '': 
            L6=3.6
            values.append(L6)
        else:
            L6=float(self.Wheelbase.text())
            values.append(L6)
            
        if self.TireRadius.text() == '': 
            L7=0.23
            values.append(L7)
        else:
            L7=float(self.TireRadius.text())
            values.append(L7)
            
        if self.MaxBHP.text() == '': 
            L8=850
            values.append(L8)
        else:
            L8=float(self.MaxBHP.text())
            values.append(L8)
            
        if self.Efficiency.text() == '': 
            L9=0.98
            values.append(L9)
        else:
            L9=float(self.Efficiency.text())
            values.append(L9)
            
        if self.StaticFriction.text() == '': 
            L10=1.7
            values.append(L10)
        else:
            L10=float(self.StaticFriction.text())
            values.append(L10)
            
        if self.CarCd.text() == '': 
            L11=0.85
            values.append(L11)
        else:
            L11=float(self.CarCd.text())
            values.append(L11)
            
        if self.CarArea.text() == '': 
            L12=1.30
            values.append(L12)
        else:
            L12=float(self.CarArea.text())
            values.append(L12)
            
        if self.RollResis.text() == '': 
            L13=0.012
            values.append(L13)
        else:
            L13=float(self.RollResis.text())
            values.append(L13)
        
        with open("Input_Parameters.txt", 'w') as output:
            for element in values:
                output.write(str(element) + '\n')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
