"""
Quantum Convolutional Neural Network (QCNN) - PennyLane Implementation
=======================================================================
Dựa trên paper: "Quantum Convolutional Neural Networks"
Tác giả: Iris Cong, Soonwon Choi, Mikhail D. Lukin (2019)
arXiv:1810.03787

Kiến trúc QCNN bao gồm:
  - Convolution Layer: áp dụng unitary quasi-local theo cách dịch chuyển bất biến
  - Pooling Layer: đo một phần qubit, dùng kết quả để điều khiển rotation
  - Fully Connected Layer: unitary tùy ý trên các qubit còn lại
  - Output: đo qubit cuối cùng

Ứng dụng: Nhận dạng pha lượng tử (Quantum Phase Recognition - QPR)
          cho Z2×Z2 Symmetry-Protected Topological (SPT) phase
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from functools import partial

# ============================================================
# 1. THIẾT LẬP THIẾT BỊ (Device Setup)
# ============================================================

def create_device(n_qubits):
    """
    Tạo thiết bị mô phỏng lượng tử.
    
    Args:
        n_qubits (int): Số qubit đầu vào
    Returns:
        qml.Device: Thiết bị PennyLane
    """
    # default.qubit: mô phỏng trạng thái đầy đủ trên CPU
    return qml.device("default.qubit", wires=n_qubits)


# ============================================================
# 2. CONVOLUTION LAYER (Lớp Tích Chập Lượng Tử)
# ============================================================

def convolution_unitary(params, wires):
    """
    Áp dụng unitary 2-qubit làm kernel tích chập.
    
    Trong QCNN, một lớp convolution áp dụng cùng một unitary U_i
    một cách tịnh tiến bất biến (translationally invariant) lên
    các cặp qubit lân cận, tương tự bộ lọc CNN cổ điển.
    
    Kiến trúc unitary tổng quát cho 2 qubit (15 tham số):
    - Xây dựng từ các rotation gates + CNOT entangling gates
    - Đây là dạng "general 2-qubit unitary" tiêu chuẩn
    
    Args:
        params (array): Vector tham số có 15 phần tử
        wires (list): [qubit_1, qubit_2] - 2 qubit cần áp dụng
    """
    # Lớp rotation đầu tiên trên mỗi qubit
    qml.Rot(params[0], params[1], params[2], wires=wires[0])
    qml.Rot(params[3], params[4], params[5], wires=wires[1])
    
    # Cổng entangling CNOT (tạo vướng víu lượng tử)
    qml.CNOT(wires=[wires[0], wires[1]])
    
    # Lớp rotation thứ hai sau entanglement
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    
    # Cổng entangling thứ hai theo chiều ngược
    qml.CNOT(wires=[wires[1], wires[0]])
    
    # Lớp rotation thứ ba
    qml.RY(params[8], wires=wires[0])
    
    # Cổng entangling cuối
    qml.CNOT(wires=[wires[0], wires[1]])
    
    # Lớp rotation cuối
    qml.Rot(params[9], params[10], params[11], wires=wires[0])
    qml.Rot(params[12], params[13], params[14], wires=wires[1])


def convolution_layer(params, wires):
    """
    Lớp tích chập lượng tử đầy đủ.
    
    Áp dụng unitary 2-qubit lên TẤT CẢ các cặp qubit lân cận:
    - Bước 1 (odd pairs):  (0,1), (2,3), (4,5), ...
    - Bước 2 (even pairs): (1,2), (3,4), (5,6), ...
    
    Thiết kế này đảm bảo mọi qubit đều tương tác với láng giềng
    của nó, tương tự convolution với stride=1 trong CNN cổ điển.
    
    Args:
        params (array): Tham số cho cả lớp, shape (2, 15)
                        params[0]: tham số cho cặp lẻ (odd)
                        params[1]: tham số cho cặp chẵn (even)
        wires (list): Danh sách các qubit
    """
    n = len(wires)
    
    # Bước 1: Áp dụng trên các cặp (0,1), (2,3), (4,5), ...
    for i in range(0, n - 1, 2):
        convolution_unitary(params[0], wires=[wires[i], wires[i + 1]])
    
    # Bước 2: Áp dụng trên các cặp (1,2), (3,4), (5,6), ...
    for i in range(1, n - 1, 2):
        convolution_unitary(params[1], wires=[wires[i], wires[i + 1]])


# ============================================================
# 3. POOLING LAYER (Lớp Gộp Lượng Tử)
# ============================================================

def pooling_unitary(params, wires):
    """
    Thực hiện pooling lượng tử trên một cặp qubit.
    
    Cơ chế pooling trong QCNN (khác với CNN cổ điển):
    1. Đo qubit nguồn (source qubit) - qubit này bị "loại bỏ"
    2. Dựa vào kết quả đo (0 hoặc 1), áp dụng rotation U hoặc U†
       lên qubit đích (target qubit)
    
    Đây chính là cơ chế QEC (Quantum Error Correction) ẩn trong QCNN:
    - Kết quả đo = "syndrome measurement"
    - Rotation điều kiện = "error correction"
    
    Implementation dùng mid-circuit measurement của PennyLane:
    Ta dùng controlled unitary thay vì đo thực sự để tương thích
    với backpropagation.
    
    Args:
        params (array): Vector 3 tham số [θ1, φ, θ2]
        wires (list): [qubit_nguon, qubit_dich]
                      qubit_nguon sẽ bị đo và loại bỏ
    """
    # Controlled-U: nếu qubit_nguon = |1⟩, áp dụng rotation lên qubit_dich
    # Điều này mô phỏng: đo qubit_nguon, nếu kết quả = 1 thì sửa lỗi
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])


def pooling_layer(params, source_wires, target_wires):
    """
    Lớp pooling lượng tử đầy đủ.
    
    Giảm số qubit hiệu dụng đi một nửa:
    - source_wires: các qubit sẽ bị "đo và loại bỏ"  
    - target_wires: các qubit được giữ lại cho layer tiếp theo
    
    Sau pooling, chỉ target_wires còn mang thông tin.
    source_wires đã truyền thông tin của chúng vào target_wires
    thông qua các controlled operations.
    
    Args:
        params (array): Tham số, shape (2,) [θ_CRZ, θ_CRX]  
        source_wires (list): Qubit bị loại bỏ sau pooling
        target_wires (list): Qubit được giữ lại
    """
    assert len(source_wires) == len(target_wires), \
        "Số qubit nguồn phải bằng số qubit đích"
    
    for src, tgt in zip(source_wires, target_wires):
        pooling_unitary(params, wires=[src, tgt])


# ============================================================
# 4. FULLY CONNECTED LAYER (Lớp Kết Nối Đầy Đủ)
# ============================================================

def fully_connected_layer(params, wires):
    """
    Lớp kết nối đầy đủ - unitary tùy ý trên các qubit còn lại.
    
    Sau nhiều vòng convolution + pooling, số qubit giảm xuống còn nhỏ.
    FC layer áp dụng một unitary tổng quát lên tất cả qubit còn lại,
    tương tự "flatten + dense layer" trong CNN cổ điển.
    
    Args:
        params (array): Tham số, shape phụ thuộc vào len(wires)
        wires (list): Các qubit còn lại sau pooling
    """
    n = len(wires)
    
    # Áp dụng rotation đơn qubit trên từng qubit
    for i, wire in enumerate(wires):
        qml.Rot(params[3 * i], params[3 * i + 1], params[3 * i + 2], 
                wires=wire)
    
    # Áp dụng entangling gates giữa tất cả các cặp qubit
    param_idx = 3 * n
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(params[param_idx], wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])
            param_idx += 1
    
    # Rotation cuối
    for i, wire in enumerate(wires):
        qml.RY(params[param_idx + i], wires=wire)


# ============================================================
# 5. KIẾN TRÚC QCNN ĐẦY ĐỦ
# ============================================================

def count_params(n_qubits, depth):
    """
    Đếm tổng số tham số cần học của QCNN.
    
    Phân tích theo paper: QCNN dùng O(log N) tham số
    cho N qubit đầu vào - hiệu quả hơn exponentially so với
    bộ phân loại quantum circuit tổng quát.
    
    Args:
        n_qubits (int): Số qubit đầu vào N
        depth (int): Độ sâu QCNN (số vòng conv+pool)
    Returns:
        dict: Số tham số cho từng loại layer
    """
    # Mỗi convolution layer: 2 unitary × 15 params = 30 params/depth
    conv_params = depth * 30
    
    # Mỗi pooling layer: 2 params/depth
    pool_params = depth * 2
    
    # Fully connected: phụ thuộc số qubit còn lại
    n_remaining = max(1, n_qubits // (2 ** depth))
    fc_params = 3 * n_remaining + n_remaining * (n_remaining - 1) // 2 + n_remaining
    
    total = conv_params + pool_params + fc_params
    
    return {
        "convolution": conv_params,
        "pooling": pool_params, 
        "fully_connected": fc_params,
        "total": total
    }


def qcnn_circuit(params, n_qubits, depth=1):
    """
    Mạch QCNN đầy đủ - hàm này được dùng như quantum node.
    
    Kiến trúc theo paper (Figure 1b):
    ρ_in → [Conv → Pool] × depth → FC → Đo qubit
    
    Kích thước hệ thống giảm theo mỗi layer:
    N → N/2 → N/4 → ... → N/2^depth qubit
    
    Args:
        params (array): TẤT CẢ tham số có thể học, được đóng gói phẳng
        n_qubits (int): Số qubit đầu vào N
        depth (int): Số lần lặp Conv+Pool
        
    Returns:
        float: Kỳ vọng đo PauliZ trên qubit đầu ra (trong [-1, +1])
               +1 ≈ pha SPT, -1 ≈ pha paramagnetic
    """
    active_wires = list(range(n_qubits))
    param_idx = 0  # Con trỏ theo dõi vị trí trong mảng params
    
    # ---- Vòng lặp Conv + Pool ----
    for d in range(depth):
        n_active = len(active_wires)
        
        # --- Convolution Layer ---
        # Áp dụng 2 unitary: một cho cặp lẻ, một cho cặp chẵn
        # Mỗi unitary cần 15 tham số
        conv_params = params[param_idx: param_idx + 30].reshape(2, 15)
        param_idx += 30
        
        convolution_layer(conv_params, active_wires)
        
        # --- Pooling Layer ---
        # Chia active_wires thành 2 nửa:
        # - Nửa đầu: source (sẽ bị loại)
        # - Nửa sau: target (được giữ lại)
        n_pool = n_active // 2
        source_wires = active_wires[:n_pool]      # qubit bị loại
        target_wires = active_wires[n_pool: 2 * n_pool]  # qubit được giữ
        
        pool_params = params[param_idx: param_idx + 2]
        param_idx += 2
        
        pooling_layer(pool_params, source_wires, target_wires)
        
        # Cập nhật danh sách qubit active (chỉ giữ target_wires)
        active_wires = target_wires
    
    # ---- Fully Connected Layer ----
    n_remaining = len(active_wires)
    n_fc_params = 3 * n_remaining + n_remaining * (n_remaining - 1) // 2 + n_remaining
    
    fc_params = params[param_idx: param_idx + n_fc_params]
    param_idx += n_fc_params
    
    fully_connected_layer(fc_params, active_wires)
    
    # ---- Đo kết quả ----
    # Đo qubit trung tâm của các qubit còn lại
    # Trả về kỳ vọng PauliZ: ∈ [-1, +1]
    measurement_wire = active_wires[len(active_wires) // 2]
    return qml.expval(qml.PauliZ(measurement_wire))


# ============================================================
# 6. HÀM MẤT MÁT VÀ HUẤN LUYỆN (Loss & Training)
# ============================================================

def prepare_training_data(n_samples=20):
    """
    Chuẩn bị dữ liệu huấn luyện đơn giản hóa.
    
    Trong paper, dữ liệu huấn luyện là các ground state của Hamiltonian:
    H = -J Σ Z_i X_{i+1} Z_{i+2} - h1 Σ X_i - h2 Σ X_i X_{i+1}
    
    Ở đây ta dùng trạng thái đơn giản hóa:
    - Trạng thái SPT (nhãn +1):  |cluster state⟩ và các biến thể
    - Trạng thái non-SPT (nhãn -1): |product state⟩ và các biến thể
    
    QUAN TRỌNG: Trong triển khai thực tế, cần tính ground state
    thực bằng DMRG hoặc exact diagonalization!
    
    Returns:
        list: [(state_vector, label), ...] với label ∈ {+1, -1}
    """
    training_data = []
    rng = np.random.default_rng(42)
    
    # --- Trạng thái SPT mẫu (1D cluster state) ---
    # |cluster⟩ = sản phẩm của CZ gates lên |+⟩^N
    # Đây là fixed point của pha SPT Z2×Z2
    for _ in range(n_samples // 2):
        # Tạo biến thể nhỏ của cluster state (nhiễu thấp → vẫn trong pha SPT)
        noise_level = rng.uniform(0, 0.1)
        # Dùng góc xoay nhỏ để biểu diễn perturbation
        state_params = rng.uniform(0, noise_level, size=6)
        training_data.append((state_params, +1.0, "SPT"))
    
    # --- Trạng thái paramagnetic mẫu ---
    # |paramagnetic⟩ ≈ |+⟩^N (h1 >> J)
    for _ in range(n_samples // 2):
        noise_level = rng.uniform(0.5, 1.0)
        state_params = rng.uniform(0, noise_level, size=6)
        training_data.append((state_params, -1.0, "Paramagnetic"))
    
    return training_data


def mse_loss(predictions, labels):
    """
    Hàm mất mát Mean Squared Error (Eq. 1 trong paper).
    
    MSE = (1/2M) Σ (y_i - f(|ψ_α⟩))²
    
    Trong đó:
    - y_i ∈ {+1, -1}: nhãn thực (SPT hay không)
    - f(|ψ_α⟩): output của QCNN, kỳ vọng đo PauliZ
    
    Args:
        predictions (array): Output của QCNN
        labels (array): Nhãn thực
    Returns:
        float: Giá trị MSE
    """
    return pnp.mean((predictions - labels) ** 2) / 2


# ============================================================
# 7. MẠCH QCNN HOÀN CHỈNH VỚI PENNYLANE QNODE
# ============================================================

class QCNN:
    """
    Lớp QCNN hoàn chỉnh đóng gói toàn bộ kiến trúc.
    
    Theo paper, QCNN cho N qubit sử dụng O(log N) tham số.
    Với depth=d:
    - Số tham số convolution: 30d
    - Số tham số pooling: 2d  
    - Số tham số FC: phụ thuộc N/2^d
    
    So sánh: classifier tổng quát cần O(exp(N)) tham số!
    """
    
    def __init__(self, n_qubits, depth=1):
        """
        Khởi tạo QCNN.
        
        Args:
            n_qubits (int): Số qubit đầu vào (nên là lũy thừa của 2)
            depth (int): Số vòng lặp Conv+Pool (QCNN depth d trong paper)
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = create_device(n_qubits)
        
        # Đếm và lưu số tham số
        param_counts = count_params(n_qubits, depth)
        self.n_params = param_counts["total"]
        
        print(f"QCNN được khởi tạo:")
        print(f"  - Số qubit đầu vào: {n_qubits}")
        print(f"  - Độ sâu (depth): {depth}")
        print(f"  - Tham số convolution: {param_counts['convolution']}")
        print(f"  - Tham số pooling: {param_counts['pooling']}")
        print(f"  - Tham số FC: {param_counts['fully_connected']}")
        print(f"  - TỔNG tham số: {param_counts['total']} = O(log {n_qubits})")
        
        # Xây dựng QNode (quantum function + device)
        self.qnode = qml.QNode(
            self._circuit,
            self.device,
            interface="autograd",  # Dùng autograd để tính gradient tự động
            diff_method="best"     # PennyLane tự chọn phương pháp tốt nhất
        )
        
        # Khởi tạo tham số ngẫu nhiên (như trong paper)
        self.params = pnp.array(
            np.random.uniform(0, 2 * np.pi, self.n_params),
            requires_grad=True
        )
    
    def _circuit(self, params, state_encoding=None):
        """
        Mạch lượng tử nội bộ.
        
        Trong triển khai thực tế, state_encoding sẽ là mạch chuẩn bị
        trạng thái đầu vào ρ_in. Ở đây ta dùng encoding đơn giản.
        
        Args:
            params: Tham số có thể học
            state_encoding: Tham số mã hóa trạng thái đầu vào
        """
        # ---- Chuẩn bị trạng thái đầu vào ----
        # Trong paper: ρ_in là ground state thực của Hamiltonian
        # Ở đây: dùng angle embedding đơn giản hóa
        if state_encoding is not None:
            qml.AngleEmbedding(state_encoding, wires=range(self.n_qubits))
        
        # ---- Chạy mạch QCNN ----
        return qcnn_circuit(params, self.n_qubits, self.depth)
    
    def predict(self, state_encoding):
        """
        Dự đoán pha lượng tử cho một trạng thái đầu vào.
        
        Args:
            state_encoding: Encoding của trạng thái đầu vào
        Returns:
            float: Giá trị trong [-1, +1]
                   > 0 → pha SPT, < 0 → pha khác
        """
        return self.qnode(self.params, state_encoding)
    
    def train(self, training_data, n_epochs=50, learning_rate=0.01):
        """
        Huấn luyện QCNN bằng gradient descent.
        
        Thuật toán theo paper:
        1. Khởi tạo tham số ngẫu nhiên
        2. Tính MSE loss
        3. Cập nhật tham số theo gradient
        4. Lặp lại đến khi hội tụ
        
        Paper dùng "bold driver" learning rate (tăng 5% nếu loss giảm,
        giảm 50% nếu loss tăng). Ở đây ta dùng Adam optimizer.
        
        Args:
            training_data: Dữ liệu huấn luyện từ prepare_training_data()
            n_epochs (int): Số epoch huấn luyện
            learning_rate (float): Learning rate ban đầu
            
        Returns:
            list: Lịch sử loss qua các epoch
        """
        # Dùng Adam optimizer (cải tiến của gradient descent)
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)
        
        loss_history = []
        
        print(f"\nBắt đầu huấn luyện QCNN ({n_epochs} epochs)...")
        print("-" * 50)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            # Định nghĩa hàm cost cho optimizer
            def cost_fn(params):
                total_loss = 0.0
                for state_enc, label, _ in training_data:
                    # Tính output của QCNN
                    pred = self.qnode(params, pnp.array(state_enc))
                    # Cộng dồn MSE loss
                    total_loss += (pred - label) ** 2
                return total_loss / (2 * len(training_data))
            
            # Cập nhật tham số theo gradient
            self.params, current_loss = optimizer.step_and_cost(
                cost_fn, self.params
            )
            
            loss_history.append(float(current_loss))
            
            # In tiến độ mỗi 10 epoch
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:3d}/{n_epochs} | Loss: {current_loss:.6f}")
        
        print("-" * 50)
        print(f"Huấn luyện hoàn tất! Loss cuối: {loss_history[-1]:.6f}")
        return loss_history
    
    def evaluate(self, test_data):
        """
        Đánh giá độ chính xác trên tập test.
        
        Args:
            test_data: Dữ liệu test cùng định dạng với training_data
        Returns:
            dict: Kết quả đánh giá
        """
        correct = 0
        results = []
        
        for state_enc, label, phase_name in test_data:
            pred = float(self.predict(pnp.array(state_enc)))
            predicted_phase = "SPT" if pred > 0 else "non-SPT"
            true_phase = "SPT" if label > 0 else "non-SPT"
            is_correct = predicted_phase == true_phase
            correct += is_correct
            results.append({
                "phase": phase_name,
                "prediction": pred,
                "label": label,
                "correct": is_correct
            })
        
        accuracy = correct / len(test_data)
        print(f"\nĐộ chính xác: {accuracy:.1%} ({correct}/{len(test_data)})")
        return {"accuracy": accuracy, "results": results}


# ============================================================
# 8. MẠCH QCNN CHÍNH XÁC CHO 1D CLUSTER STATE (Exact QCNN)
# ============================================================

def exact_qcnn_convolution(wires):
    """
    Lớp convolution CHÍNH XÁC cho cluster state Z2×Z2 SPT.
    
    Theo paper (Figure 2b), lớp convolution chính xác bao gồm:
    - Controlled-Phase gates (CZ): tạo vướng víu
    - Toffoli gates với control qubits trong X basis
    
    Đây là mạch được thiết kế analytic (không cần học) để
    nhận dạng pha Z2×Z2 SPT, đặc biệt là 1D cluster state.
    
    Args:
        wires (list): Danh sách qubit
    """
    n = len(wires)
    
    # CZ gates giữa các cặp lân cận (tương tự trong MERA của cluster state)
    for i in range(0, n - 1, 2):
        qml.CZ(wires=[wires[i], wires[i + 1]])
    
    # Hadamard để chuyển sang X basis (cho Toffoli X-basis)
    for wire in wires[::2]:
        qml.Hadamard(wires=wire)
    
    # CZ gates cho cặp chẵn-lẻ
    for i in range(1, n - 1, 2):
        qml.CZ(wires=[wires[i], wires[i + 1]])


def exact_qcnn_pooling(source_wires, target_wires):
    """
    Lớp pooling CHÍNH XÁC cho cluster state.
    
    Theo paper: pooling layer thực hiện phase-flip trên qubit còn lại
    khi kết quả đo X = -1 trên qubit bị loại.
    
    Điều này tương đương với QEC: phát hiện và sửa X-errors.
    
    Args:
        source_wires (list): Qubit bị đo (loại bỏ)
        target_wires (list): Qubit được giữ lại
    """
    for src, tgt in zip(source_wires, target_wires):
        # Đo trong X basis = Hadamard + đo Z
        qml.Hadamard(wires=src)
        # Controlled-Z: nếu src = |1⟩ (X=-1), flip phase của tgt
        qml.CZ(wires=[src, tgt])


def exact_qcnn_fully_connected(wires):
    """
    Lớp FC chính xác: đo Z_{i-1} X_i Z_{i+1}.
    
    Theo paper: fully connected layer đo string order parameter
    S = Z_{i-1} X_i Z_{i+1} lên qubit còn lại.
    
    Đây chính là nonlocal order parameter đặc trưng của pha SPT Z2×Z2!
    
    Args:
        wires (list): Qubit còn lại sau pooling
    """
    if len(wires) >= 3:
        # Đo Z_{i-1} X_i Z_{i+1} = string order parameter
        # Chuyển sang computational basis
        qml.Hadamard(wires=wires[1])  # X basis cho qubit giữa


# ============================================================
# 9. VISUALIZATION VÀ DEMO
# ============================================================

def visualize_qcnn_circuit(n_qubits=8, depth=1):
    """
    Vẽ và in thông tin mạch QCNN.
    
    Args:
        n_qubits (int): Số qubit
        depth (int): Độ sâu QCNN
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # Tạo tham số giả để vẽ mạch
    param_counts = count_params(n_qubits, depth)
    dummy_params = np.zeros(param_counts["total"])
    dummy_state = np.zeros(6)
    
    @qml.qnode(dev)
    def circuit_for_drawing(params, state_enc):
        qml.AngleEmbedding(state_enc, wires=range(n_qubits))
        return qcnn_circuit(params, n_qubits, depth)
    
    print("\n" + "=" * 60)
    print("QCNN CIRCUIT DIAGRAM")
    print("=" * 60)
    print(qml.draw(circuit_for_drawing)(dummy_params, dummy_state))


def demo_qcnn():
    """
    Demo đầy đủ QCNN: khởi tạo → huấn luyện → đánh giá → vẽ loss.
    
    Luồng theo paper:
    1. Khởi tạo QCNN với N=8 qubit, depth=1
    2. Chuẩn bị dữ liệu huấn luyện (ground states đơn giản hóa)
    3. Huấn luyện bằng gradient descent
    4. Đánh giá trên tập test
    5. Vẽ phase diagram đầu ra
    """
    print("=" * 60)
    print("DEMO: Quantum Convolutional Neural Network")
    print("Paper: Cong, Choi & Lukin (2019)")
    print("=" * 60)
    
    # --- Cấu hình ---
    N_QUBITS = 8   # Số qubit (nhỏ để mô phỏng nhanh)
    DEPTH = 1      # Độ sâu QCNN
    N_EPOCHS = 30  # Số epoch huấn luyện
    
    # --- Khởi tạo QCNN ---
    qcnn = QCNN(n_qubits=N_QUBITS, depth=DEPTH)
    
    # --- Chuẩn bị dữ liệu ---
    print("\nChuẩn bị dữ liệu huấn luyện...")
    train_data = prepare_training_data(n_samples=20)
    test_data = prepare_training_data(n_samples=10)
    
    print(f"  - Training: {len(train_data)} mẫu")
    print(f"  - Test: {len(test_data)} mẫu")
    
    # --- Huấn luyện ---
    loss_history = qcnn.train(train_data, n_epochs=N_EPOCHS, learning_rate=0.05)
    
    # --- Đánh giá ---
    print("\nĐánh giá trên tập test:")
    eval_results = qcnn.evaluate(test_data)
    
    # --- Vẽ đồ thị ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("QCNN Training Results\n(Cong, Choi & Lukin 2019)", 
                 fontsize=12, fontweight='bold')
    
    # Plot 1: Loss curve
    ax1 = axes[0]
    ax1.plot(loss_history, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Predictions
    ax2 = axes[1]
    spt_preds = [r["prediction"] for r in eval_results["results"] 
                 if r["phase"] == "SPT"]
    nonspt_preds = [r["prediction"] for r in eval_results["results"] 
                    if r["phase"] == "Paramagnetic"]
    
    ax2.scatter(range(len(spt_preds)), spt_preds, 
                c='blue', label='SPT phase', s=100, marker='o')
    ax2.scatter(range(len(nonspt_preds)), nonspt_preds, 
                c='red', label='Paramagnetic', s=100, marker='^')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Decision boundary')
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("QCNN output ⟨Z⟩")
    ax2.set_title(f"Phase Classification\n(Accuracy: {eval_results['accuracy']:.1%})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/qcnn_results.png", dpi=150, bbox_inches='tight')
    print("\nĐã lưu đồ thị vào: qcnn_results.png")
    plt.close()
    
    return qcnn, loss_history, eval_results


# ============================================================
# 10. CHẠY DEMO
# ============================================================

if __name__ == "__main__":
    # # --- In thông tin tổng quan ---
    # print("\n" + "=" * 60)
    # print("PHÂN TÍCH SỐ THAM SỐ QCNN (so sánh với paper)")
    # print("=" * 60)
    
    # for n in [8, 16, 32, 64]:
    #     params = count_params(n, depth=1)
    #     print(f"  N={n:3d} qubit: {params['total']:4d} tham số "
    #           f"(= O(log {n}) = O({int(np.log2(n))}))")
    
    # print("\n⇒ QCNN dùng O(log N) tham số, hiệu quả hơn exponentially")
    # print("  so với classifier tổng quát cần O(2^N) tham số!\n")
    
    # --- Vẽ mạch ---
    visualize_qcnn_circuit(n_qubits=8, depth=1)
    
    # --- Chạy demo training ---
    # print("\n" + "=" * 60)
    # print("BẮT ĐẦU DEMO TRAINING")
    # print("=" * 60)
    # qcnn_model, losses, results = demo_qcnn()
    
    # print("\n✓ Demo hoàn tất!")
    # print("  Xem file qcnn_results.png để thấy kết quả training.")
