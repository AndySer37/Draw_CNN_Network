
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *
# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input('../img/hunts.png' ,  name="input", width=9, height=6),
    #block-001
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(0,0,0)", width=1, height=20, depth=40, opacity=0.5),
    to_Conv( name='ccr_b1', s_filer="16 * 50", n_filer=64, offset="(0,0,0)", to="(pool_b1-east)", width=2, height=20, depth=40 ,caption="conv1"),

    to_Pool(name="pool_b2", offset="(1.5,0,0)", to="(ccr_b1-east)", width=1, height=15, depth=30, opacity=0.5),
    to_Conv( name='ccr_b2', s_filer="8 * 25", n_filer=128, offset="(0,0,0)", to="(pool_b2-east)", width=2, height=15, depth=30 ,caption="conv2"),

    to_Pool(name="pool_b3", offset="(1.5,0,0)", to="(ccr_b2-east)", width=1, height=10, depth=20, opacity=0.5),
    to_Conv( name='ccr_b3_1', s_filer="", n_filer="", offset="(0,0,0)", to="(pool_b3-east)", width=2, height=10, depth=20 ),
    to_Conv( name='ccr_b3_2', s_filer="", n_filer="128", offset="(0,0,0)", to="(ccr_b3_1-east)", width=2, height=10, depth=20 ,caption="conv3"),
    to_Conv( name='ccr_b3_3', s_filer="4 * 12", n_filer="", offset="(0,0,0)", to="(ccr_b3_2-east)", width=2, height=10, depth=20 ),

    to_Pool(name="pool_b4", offset="(1.5,0,0)", to="(ccr_b3_3-east)", width=1, height=8, depth=16, opacity=0.5,caption="pool"),
    to_SoftMax(name="tanh", s_filer="3 * 11", offset="(1,0,0)", to="(pool_b4-east)", width=1, height=8, depth=16, opacity=0.5,caption="tanh"),
    to_UnPool( name='dconv1', offset="(1.5,0,0)", to="(tanh-east)", width=2, height=20, depth=40,caption="Offset Map"),

    to_connection( "ccr_b1", "pool_b2"),
    to_connection( "ccr_b2", "pool_b3"),
    to_connection( "ccr_b3_3", "pool_b4"),
    to_connection( "pool_b4", "tanh"),
    to_connection( "tanh", "dconv1"),

    to_Conv( name='ccr_b4', s_filer="32 * 100", n_filer=64, offset="(1.5,0,0)", to="(dconv1-east)", width=2, height=20, depth=40 ,caption="conv4"),
    to_Pool(name="pool_b5", offset="(1,0,0)", to="(ccr_b4-east)", width=1, height=15, depth=30, opacity=0.5),
    to_Conv( name='ccr_b5', s_filer="16 * 50", n_filer=128, offset="(1.5,0,0)", to="(pool_b5-east)", width=2, height=15, depth=30 ,caption="conv5"),
    to_Pool(name="pool_b6", offset="(1,0,0)", to="(ccr_b5-east)", width=1, height=10, depth=20, opacity=0.5),
    to_Conv( name='ccr_b6_1', s_filer="", n_filer="", offset="(1.5,0,0)", to="(pool_b6-east)", width=2, height=10, depth=20 ,caption=""),
    to_Conv( name='ccr_b6_2', s_filer="8 * 25", n_filer=256, offset="(0,0,0)", to="(ccr_b6_1-east)", width=2, height=10, depth=20 ,caption="conv6"),
    to_Pool(name="pool_b7", offset="(1,0,0)", to="(ccr_b6_2-east)", width=1, height=8, depth=16, opacity=0.5),
    to_Conv( name='ccr_b7_1', s_filer="", n_filer="", offset="(1.5,0,0)", to="(pool_b7-east)", width=2, height=8, depth=16 ,caption=""),
    to_Conv( name='ccr_b7_2', s_filer="4 * 25", n_filer=512, offset="(0,0,0)", to="(ccr_b7_1-east)", width=2, height=8, depth=16 ,caption="conv7"),
    to_Pool(name="pool_b8", offset="(1,0,0)", to="(ccr_b7_2-east)", width=1, height=6, depth=12, opacity=0.5),
    to_Conv( name='ccr_b8', s_filer="1*25", n_filer=512, offset="(1,0,0)", to="(pool_b8-east)", width=1, height=3, depth=6 ,caption="conv8"),
    to_SoftMax(name="blstm1", s_filer="", offset="(1.5,0,0)", to="(ccr_b8-east)", width=2, height=3, depth=6, opacity=0.5,caption="BLSTM"),
    to_SoftMax(name="blstm2", s_filer="1*25", offset="(0,0,0)", to="(blstm1-east)", width=2, height=3, depth=6, opacity=0.5,caption=""),
    to_SoftMax(name="gru", s_filer="1*25", offset="(1,0,0)", to="(blstm2-east)", width=2, height=3, depth=6, opacity=0.5,caption="GRU"),
    
    to_connection( "dconv1", "ccr_b4"),
    to_connection( "pool_b5", "ccr_b5"),
    to_connection( "pool_b6", "ccr_b6_1"),
    to_connection( "pool_b7", "ccr_b7_1"),
    to_connection( "pool_b8", "ccr_b8"),
    to_connection( "ccr_b8", "blstm1"),
    to_connection( "blstm2", "gru"), 

    to_Conv("senet_1", 1, 64, offset="(2,0,15)", to="(ccr_b4-east)", height=1, depth=1, width=6 ,caption="SE Block 1"),
    to_connection( "ccr_b4", "senet_1"),
    to_connection( "senet_1", "pool_b5"),
    to_connection( "ccr_b4", "pool_b5"), 

    to_Conv("senet_2", 1, 128, offset="(2,0,13)", to="(ccr_b5-east)", height=1, depth=1, width=6 ,caption="SE Block 2"),
    to_connection( "ccr_b5", "senet_2"),
    to_connection( "senet_2", "pool_b6"),
    to_connection( "ccr_b5", "pool_b6"), 

    to_Conv("senet_3", 1, 256, offset="(2,0,11)", to="(ccr_b6_2-east)", height=1, depth=1, width=6 ,caption="SE Block 3"),
    to_connection( "ccr_b6_2", "senet_3"),
    to_connection( "senet_3", "pool_b7"),
    to_connection( "ccr_b6_2", "pool_b7"), 

    to_Conv("senet_4", 1, 512, offset="(2,0,10)", to="(ccr_b7_2-east)", height=1, depth=1, width=6 ,caption="SE Block 4"),
    to_connection( "ccr_b7_2", "senet_4"),
    to_connection( "senet_4", "pool_b8"),
    to_connection( "ccr_b7_2", "pool_b8"), 

    to_input('../img/word.png' ,to="(30.2,0,0)", name="output", width=6, height=4),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
