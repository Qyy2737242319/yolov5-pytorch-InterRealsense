from tkinter import *
def callback():
    ceshi()
GUI=Tk()
GUI.title('机器人控制台')
GUI.geometry('600x400')
Button1=Button(GUI,text='开始识别箭',command=callback)
Button1.pack()
GUI.mainloop()