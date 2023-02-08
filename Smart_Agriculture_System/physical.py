import sys
from threading import Thread
from Adafruit_IO import MQTTClient
import time
import serial.tools.list_ports
import keyboard

import utils.read_sensors as read_sensors
import utils.relay_control as relay_control
import utils.port as port

import cv2


#Constants
AIO_FEED_IDs = ["sensor1", "sensor2", "sensor3", "sensor4", "relay1", "relay2", "AI"]
AIO_USERNAME = "Fusioz"
AIO_KEY = "aio_HtYs87cVQ6F4VtA4wZkIlaHbl0lv"
PUBLISH_INTERVAL = 50
DETECTION_INTERVAL = 100

#Variables
count = 1
count_detect = 1
isRelay1Signal = False
isRelay1 = False
isRelay2Signal = False
isRelay2 = False
analysis_light = None
day_of_growth = 0

ser = None

#Connect to serial port
portName = port.getPort()
print("Port: " + portName)
if portName != "None":
    ser = serial.Serial(port=portName, baudrate=9600)

#MQTt callbacks
def connected(client):
    print("Ket noi thanh cong ...")
    for topic in AIO_FEED_IDs:
        client.subscribe(topic)

def subscribe(client , userdata , mid , granted_qos):
    print("Subscribe thanh cong...")

def disconnected(client):
    print("Ngat ket noi ...")
    sys.exit (1)

def message(client , feed_id , payload):
    #Update relay states based on MQTT messages
    
    if feed_id == "relay1":
        global isRealay1Signal, isRelay1
        isRealay1Signal = True
        if payload == "ON":
            isRelay1 = True
            print("Relay 1: ON")
        else:
            isRelay1 = False
            print("Relay 1: OFF")
    elif feed_id == "relay2":
        global isRelay2Signal, isRelay2
        isRelay2Signal = True
        if payload == "ON":
            isRelay2 = True
            print("Relay 2: ON")
        else:
            isRelay2 = False
            print("Relay 2: OFF")

#Connect to Adafruit IO
client = MQTTClient(AIO_USERNAME , AIO_KEY)
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message
client.on_subscribe = subscribe
client.connect()
client.loop_background()

#Main loop
def physical_device_loop():
    global count
    global isRelay1Signal
    global isRelay1
    global isRelay2Signal
    global isRelay2
    global analysis_light
    global day_of_growth
    update_light_counter = 60
    while True:
        if count == PUBLISH_INTERVAL:
            air_temp_value = read_sensors.readTemperature(ser, port.serial_read_data)/10
            print(f"Air Temperature: {air_temp_value}°C")
            client.publish("sensor2", air_temp_value)
            air_humi_value = read_sensors.readMoisture(ser, port.serial_read_data)/10
            print(f"Air Humidity: {air_humi_value}%")
            client.publish("sensor1", air_humi_value)

            soil_temp_value = read_sensors.readSoilTemp(ser, port.serial_read_data)/100
            print(f"Soil Temperature: {soil_temp_value}°C")
            client.publish("sensor3", soil_temp_value)
            soil_humi_value = read_sensors.readSoilMoisture(ser, port.serial_read_data)/100
            print(f"Soil Humidity: {soil_humi_value}%")
            client.publish("sensor4", soil_humi_value)
            count = 0
        if isRelay1Signal:
            if isRelay1:
                relay_control.setDevice1(True, ser)
            else:
                relay_control.setDevice1(False, ser)
            isRelay1Signal = False
        if isRelay2Signal:
            if isRelay2:
                relay_control.setDevice2(True, ser)
            else:
                relay_control.setDevice2(False, ser)
            isRelay2Signal = False
        if keyboard.is_pressed('u'):
            try:
                from data_analysis import light
                analysis_light = light
            except ImportError:
                print("There is no function !")
        
        if analysis_light != None :
            update_light_counter -= 1
            if update_light_counter == 0:
                day = analysis_light(1)
                if day != None:
                    day_of_growth += day
                print(day_of_growth)
                update_light_counter = 60
                
        count += 1
        time.sleep(0.5)