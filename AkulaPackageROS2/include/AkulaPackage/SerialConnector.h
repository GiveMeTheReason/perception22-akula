//
// Created by devilox on 30.04.2021.
//
//-----------------------------//
#ifndef SERIALCONNECTOR_H
#define SERIALCONNECTOR_H
//-----------------------------//
#include <iostream>
#include <cstring>
//-----------------------------//
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
//-----------------------------//
class SerialConnector {
public:
    SerialConnector(const std::string& tSerialPath, uint16_t tSpeed, uint32_t tTimeout, size_t tPacketSize);
    ~SerialConnector();

    //----------//

    ssize_t readSerial(uint8_t* tBuffer) const;
    ssize_t writeSerial(const uint8_t* tBuffer) const;
private:
    int         mSerialPort     = 0;
    termios     mTTY            = {};

    size_t      mPacketSize     = 0;
    uint32_t    mTimeoutMs      = 0;

    //----------//

    bool checkNewData() const;
};
//-----------------------------//
#endif
