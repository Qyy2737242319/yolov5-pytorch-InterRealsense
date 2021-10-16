import serial as ser
import struct,time
se=ser.Serial("/dev/ttyTHS1",9600,timeout=1)#第一个参数是nano串口号，第二个是波特率，第三个是传输数据允许的时间间隔
se.write("666".encode("GB2312")#第一个是要发的数据，需要是字符串，第二个是编码格式，默认是GB2312
