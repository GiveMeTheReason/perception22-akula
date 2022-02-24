//
// Created by mrob on 1/12/22.
//
//-----------------------------//
#include <rclcpp/rclcpp.hpp>
//-----------------------------//
#include "dSocket/dSocket.h"
#include "akula_package/msg/encoders.hpp"
//-----------------------------//
/**
 * This node gets encoder values from Akula desktop server via UDP socket
 * and publishes the values to the <b>encoders</b> topic
 */
class AkulaEncoderNode : public rclcpp::Node {
public:
    /**
     * Constructor creates publisher and sets up UDP socket for obtaining data from
     * the desktop server
     */
    AkulaEncoderNode() : Node("AkulaEncoderNode") {
        dSocketResult SocketRes;

        mEncoders   = create_publisher <akula_package::msg::Encoders>("encoders", 1000);
        mSocketUDP  = new dSocket(true);

        if ((SocketRes = mSocketUDP -> init(dSocketProtocol::UDP)) != dSocketResult::SUCCESS) {
            switch (SocketRes) {
                case dSocketResult::WRONG_PROTOCOL:
                    throw std::runtime_error("Server::Server: WRONG_PROTOCOL");
                case dSocketResult::CREATE_FAILURE:
                    throw std::runtime_error("Server::Server: CREATE_FAILURE, " + mSocketUDP -> getLastError());
                default:
                    throw std::runtime_error("Server::Server: this should have not occurred!");
            }
        }

        if ((SocketRes = mSocketUDP -> finalize(dSocketType::SERVER, 50001)) != dSocketResult::SUCCESS) {
            switch (SocketRes) {
                case dSocketResult::WRONG_PROTOCOL:
                    throw std::runtime_error("Server::Server: WRONG_PROTOCOL");
                case dSocketResult::NO_SOCKET_TYPE:
                    throw std::runtime_error("Server::Server: NO_SOCKET_TYPE");
                case dSocketResult::BIND_FAILURE:
                    throw std::runtime_error("Server::Server: BIND_FAILURE, " + mSocketUDP -> getLastError());
                case dSocketResult::LISTEN_FAILURE:
                    throw std::runtime_error("Server::Server: LISTEN_FAILURE, " + mSocketUDP -> getLastError());
                case dSocketResult::ADDRESS_CONVERSION_FAILURE:
                    throw std::runtime_error("Server::Server: ADDRESS_CONVERSION_FAILURE, " + mSocketUDP -> getLastError());
                default:
                    throw std::runtime_error("Server::Server: this should have not occurred!");
            }
        }
    }

    ~AkulaEncoderNode() {
        if (mSocketUDP) {
            delete(mSocketUDP);
            mSocketUDP = nullptr;
        }
    }

    /**
     * Main loop blocks on the <b>readUDP</b> function, obtains data and publishes it to
     * the <b>encoders</b> topic
     * @return Returns any occurred socket read errors
     */
    dSocketResult startLoop() {
        ///---TODO: move the size to a config---///
        size_t PacketSize = 32;
        auto Packet = new uint8_t[PacketSize];
        sockaddr_in ClientAddr {};
        socklen_t ClientAddrLen = sizeof(ClientAddr);
        ssize_t ReadBytes;
        size_t ReadTotal = 0;
        dSocketResult Result;

        while (rclcpp::ok()) {
            while (ReadTotal < PacketSize && rclcpp::ok()) {
                Result = mSocketUDP -> readUDP(Packet + ReadTotal, PacketSize - ReadTotal, &ReadBytes,
                                               reinterpret_cast <sockaddr*>(&ClientAddr), &ClientAddrLen);

                switch (Result) {
                    case dSocketResult::SUCCESS:
                        ReadTotal += ReadBytes;
                        break;
                    case dSocketResult::RECV_TIMEOUT:
                        ReadTotal = 0;
                        break;
                    default:
                        ///---TODO: skip some particular errors and terminate on others---///
                        delete[](Packet);
                        return dSocketResult::READ_ERROR;
                }
            }

            ReadTotal = 0;

            if (!rclcpp::ok()) {
                break;
            }

            akula_package::msg::Encoders Msg;
            uint32_t StmTime;
            memcpy(&StmTime, Packet + 4, 4);

            Msg.header.frame_id = "encoders";
            Msg.header.stamp.sec = StmTime / 1000;
            Msg.header.stamp.nanosec = StmTime % 1000 * 1000;

            memcpy(&Msg.left, Packet + 8, 4);
            memcpy(&Msg.right, Packet + 12, 4);

            mEncoders -> publish(Msg);
        }

        delete[](Packet);

        return dSocketResult::SUCCESS;
    }
private:
    dSocket*                                                                mSocketUDP                  = nullptr;
    std::shared_ptr <rclcpp::Publisher <akula_package::msg::Encoders>>      mEncoders;
};
//-----------------------------//
void sigintHandler(int tSigNum) {
    std::cout << "Received signum: " << tSigNum << std::endl;
    rclcpp::shutdown();
}
//-----------------------------//
int main(int argc, char ** argv) {
    signal(SIGINT, sigintHandler);
    rclcpp::init(argc, argv);

    try {
        auto EncoderNode = std::make_shared <AkulaEncoderNode>();
        if (EncoderNode -> startLoop() == dSocketResult::READ_ERROR) {
            throw std::runtime_error("Read error!");
        }
    } catch (const std::exception& tExcept) {
        std::cerr << tExcept.what() << std::endl;
    }

    rclcpp::shutdown();

    return EXIT_SUCCESS;
}
