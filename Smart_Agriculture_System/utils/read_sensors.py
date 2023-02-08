import time

print("Test Sensors")

air_temperature =[3, 3, 0, 0, 0, 1, 133, 232]
def readTemperature(ser, serial_read_data):
    serial_read_data(ser)
    ser.write(air_temperature)
    time.sleep(0.5)
    return serial_read_data(ser)

air_moisture = [3, 3, 0, 1, 0, 1, 212, 40]
def readMoisture(ser, serial_read_data):
    serial_read_data(ser)
    ser.write(air_moisture)
    time.sleep(0.5)
    return serial_read_data(ser)

soil_temperature = [1, 3, 0, 6, 0, 1, 100, 11]
def readSoilTemp(ser, serial_read_data):
    serial_read_data(ser)
    ser.write(soil_temperature)
    time.sleep(0.5)
    return serial_read_data(ser)

soil_moisture = [1, 3, 0, 7, 0, 1, 53, 203]
def readSoilMoisture(ser, serial_read_data):
    serial_read_data(ser)
    ser.write(soil_moisture)
    time.sleep(0.5)
    return serial_read_data(ser)






















