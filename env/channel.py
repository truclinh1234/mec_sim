# =============================================================================
# env/channel.py — Channel Model

#   Băng thông chia đều cho số user đang offload cùng lúc.
#   effective_rate = CHANNEL_RATE_BPS / num_concurrent_tx
#
# =============================================================================
import math
import config as cfg


class BaseChannelModel:
    """Interface chung. Override compute_rate() để thay model."""

    def compute_rate(self, user_id: int, edge_id: int,
                     num_concurrent: int) -> float:
        """
        Trả về tốc độ kênh hiệu dụng (bits/s) cho 1 user.

        Params:
            user_id        : user đang offload
            edge_id        : edge nhận task
            num_concurrent : số user đang offload cùng time-step này
        """
        raise NotImplementedError


class SharedBandwidthChannel(BaseChannelModel):
    """
    Chia đều băng thông cho tất cả user đang offload cùng lúc.

    Giả định: tất cả user dùng chung 1 kênh đến edge.
    effective_rate = CHANNEL_RATE_BPS / num_concurrent
    """

    def compute_rate(self, user_id: int, edge_id: int,
                     num_concurrent: int) -> float:
        n = max(1, num_concurrent)
        return cfg.CHANNEL_RATE_BPS / n


class FixedChannel(BaseChannelModel):
    """
    Tốc độ cố định cho mỗi user, không phụ thuộc tải.
    Dùng để so sánh baseline hoặc khi kênh là dedicated.
    """

    def compute_rate(self, user_id: int, edge_id: int,
                     num_concurrent: int) -> float:
        return cfg.CHANNEL_RATE_BPS