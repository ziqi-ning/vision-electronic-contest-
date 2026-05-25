"""test_serial_client.py — 串口客户端单元测试（mock）"""

import pytest
from unittest.mock import MagicMock, patch


class TestSerialClient:
    def test_creation(self):
        with patch("src.comm.serial_client.serial") as mock_serial:
            from src.comm.serial_client import SerialClient
            client = SerialClient()
            assert client._port == "/dev/ttyAMA4"
            assert client._baudrate == 256000
            assert client._timeout == 0.5

    def test_open_success(self):
        with patch("src.comm.serial_client.serial") as mock_serial:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial.Serial.return_value = mock_instance

            from src.comm.serial_client import SerialClient
            client = SerialClient()
            result = client.open()
            assert result is True

    def test_open_failure(self):
        with patch("src.comm.serial_client.serial") as mock_serial:
            mock_serial.Serial.side_effect = Exception("Port not found")
            from src.comm.serial_client import SerialClient
            client = SerialClient()
            result = client.open()
            assert result is False

    def test_is_open_property(self):
        with patch("src.comm.serial_client.serial") as mock_serial:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial.Serial.return_value = mock_instance

            from src.comm.serial_client import SerialClient
            client = SerialClient()
            assert client.is_open is False  # 初始未 open
            client.open()
            assert client.is_open is True

    def test_work_mode_property(self):
        with patch("src.comm.serial_client.serial"):
            from src.comm.serial_client import SerialClient
            client = SerialClient()
            assert client.work_mode == 1  # ModeCtrl 默认 1
