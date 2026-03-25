import torch
import torch.nn as nn
import torchvision.models as models 
import pennylane as qml 


class Residual_block (nn.Module) : 
    def __init__ ( self, input, output ) : 
        super ( Residual_block,self ).__init__() 
        self.input = input 
        self.output = output 
        self.conv1  = nn.Conv2d(in_channels=3, kernel_size= (3,3),  stride = 1, out_channels= 16 , padding =1  )
        self.nom1 = nn.BatchNorm2d(num_features=16)
        self.act1 = nn.ReLU(inplace=True ) 
        self.conv2 = nn.Conv2d(in_channels=16, kernel_size =(3,3),stride=1, padding= 1 , out_channels = 64   )
        self.nom2= nn.BatchNorm2d (64)
        self.shortcut = nn.Sequential() 
        if input != output: 
            self.shortcut = nn.Sequential( 
                nn.Conv2d(in_channels=input, out_channels=output, kernel_size=1, padding=0),
                nn.BatchNorm2d(output)
            )
    def forward ( self, x ) :     
        identity = self.shortcut(x) 

        out = self.conv1(x)
        out = self.nom1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.nom2(out)

        out += identity 
        out = self.act1(out)
        return out
    

 
def vqc_block ( n_qubits : int ) : 
    dev = qml.device ("default.qubit", wires = n_qubits  )
    @qml.qnode ( dev, interface = "torch")
    def circuit ( input, weight ) : 
        for i in range (n_qubits) :
            qml.Hadamard ( wires = i )

        qml.AngleEmbedding ( input, wires = range ( n_qubits))
        for wire in range (n_qubits) : 
            qml.RY (weight [0,wire], wires = wire)
            qml.RZ ( weight[1, wire], wires = wire )
        for i in range  (n_qubits ) : 
            qml.CNOT(wire=[i, i+1 ])    
        qml.CNOT ( wire =[n_qubits-1,0 ])
        return    [qml.expval ( qml.PauliZ(w)) for w in range ( n_qubits )]
    weight_shapes = {'Weigth':(2,n_qubits)}
    return qml.qnn.TorchLayer (circuit, weight_shapes) 
    
class QuantumGate(nn.Module):
    def __init__(self, output_dim: int, n_qubits: int):
        super().__init__()
        
        self.vqc = vqc_block(n_qubits)
        self.output_linear = nn.Linear(n_qubits, output_dim)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        

        
        
        q_features = self.vqc(x_t)
        return self.output_linear(1)

class Pre_model (nn.Module ) : 
    def __init__ ( self, qubits ) : 
        super(Pre_model, self ).__init__ () 
        self.residual = Residual_block (input = 3, output= 64 ) 
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear (64, qubits )
        self.tanh = nn.Tanh()
        
    def forward ( self, x ) :  
        x = self.residual(x)  
        x = self.pool(x)
        x= torch.flatten (x,1 )
        x = self.fc(x)
        x= self.tanh(x)
        return x 
    
model = Pre_model(4)    

dummy_input = torch.randn(1, 3, 28, 28) 

def check_shapes(model, input_data):
    print(f"--- Bắt đầu kiểm tra luồng dữ liệu ---")
    print(f"1. Đầu vào (Ảnh thô): {input_data.shape}")
    
    
    x = model.residual(input_data)
    print(f"2. Sau Residual Block (64 kênh): {x.shape}")
    
  
    x = model.pool(x)
    print(f"3. Sau AdaptiveAvgPool2d: {x.shape}")
    
 
    x = torch.flatten(x, 1)
    print(f"4. Sau khi Flatten (để vào Linear): {x.shape}")
    
   
    x = model.fc(x)
    x = model.tanh(x)
    print(f"5. Đầu ra cuối cùng cho 4 Qubits: {x.shape}")
    print(f"--- Hoàn tất ---")

check_shapes(model, dummy_input)

        