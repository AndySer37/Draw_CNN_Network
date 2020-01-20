
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *
# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input('../img/TS_img.jpg' ,  name="input"),
    #block-001
    to_ConvConvRelu( name='ccr_b1', s_filer=512, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40 ,caption="conv1"),
    to_Pool(name="pool_b1", offset="(2,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5, caption="pool1"),
    to_ConvConvRelu( name='ccr_b2', s_filer=256, n_filer=(128,128), offset="(2,0,0)", to="(pool_b1-east)", width=(3,3), height=32, depth=32 ,caption="conv2"),
    to_Pool(name="pool_b2", offset="(2,0,0)", to="(ccr_b2-east)", width=1, height=25, depth=25, opacity=0.5, caption="pool2"),
    to_Conv("ccr_b3", 128, 64, offset="(2,0,0)", to="(pool_b2-east)", height=25, depth=25, width=3 ),
    to_ConvConvRelu( name='ccr_b3_2', s_filer=128, n_filer=(256,256), offset="(0,0,0)", to="(ccr_b3-east)", width=(3.5,3.5), height=25, depth=25 ,caption="conv3"),
    to_Pool(name="pool_b3", offset="(2,0,0)", to="(ccr_b3_2-east)", width=1, height=16, depth=16, opacity=0.5, caption="pool3"),
    to_Conv("ccr_b4", 64, 64, offset="(2,0,0)", to="(pool_b3-east)", height=16, depth=16, width=3.5 ),
    to_ConvConvRelu( name='ccr_b4_2', s_filer=64, n_filer=(512,512), offset="(0,0,0)", to="(ccr_b4-east)", width=(4,4), height=16, depth=16 ,caption="conv4"),
    to_Pool(name="pool_b4", offset="(1.5,0,0)", to="(ccr_b4_2-east)", width=1, height=9, depth=9, opacity=0.5, caption="pool4"),
    to_Conv("ccr_b5", 32, 64, offset="(2,0,0)", to="(pool_b4-east)", height=9, depth=9, width=4 ),
    to_ConvConvRelu( name='ccr_b5_2', s_filer=32, n_filer=(512,512), offset="(0,0,0)", to="(ccr_b5-east)", width=(4,4), height=9, depth=9 ,caption="conv5"),
    to_Pool(name="pool_b5", offset="(1.5,0,0)", to="(ccr_b5_2-east)", width=1, height=4, depth=4, opacity=0.5, caption="pool5"),
    
    to_Conv("senet_1", 1, 64, offset="(0,0,0)", to="(3,0,20)", height=1, depth=1, width=10 ,caption="SE Block 1"),
    to_connection( "ccr_b1", "senet_1"),
    to_connection( "senet_1", "pool_b1"),
    to_connection( "ccr_b1", "pool_b1"), 

    to_Conv("senet_2", 1, 128, offset="(0,0,0)", to="(8,0,18)", height=1, depth=1, width=10 ,caption="SE Block 2"),
    to_connection( "ccr_b2", "senet_2"),
    to_connection( "senet_2", "pool_b2"),
    to_connection( "ccr_b2", "pool_b2"), 

    to_Conv("senet_3", 1, 256, offset="(0,0,0)", to="(15,0,16)", height=1, depth=1, width=10 ,caption="SE Block 3"),
    to_connection( "ccr_b3_2", "senet_3"),
    to_connection( "senet_3", "pool_b3"),
    to_connection( "ccr_b3_2", "pool_b3"), 

    to_Conv("senet_4", 1, 512, offset="(0,0,0)", to="(21,0,14)", height=1, depth=1, width=10 ,caption="SE Block 4"),
    to_connection( "ccr_b4_2", "senet_4"),
    to_connection( "senet_4", "pool_b4"),
    to_connection( "ccr_b4_2", "pool_b4"), 

    to_Conv("senet_5", 1, 512, offset="(0,0,0)", to="(28,0,12)", height=1, depth=1, width=10 ,caption="SE Block 5"),
    to_connection( "ccr_b5_2", "senet_5"),
    to_connection( "senet_5", "pool_b5"),
    to_connection( "ccr_b5_2", "pool_b5"), 

    to_connection( "pool_b1", "ccr_b2"), 
    to_connection( "pool_b2", "ccr_b3"), 
    to_connection( "pool_b3", "ccr_b4"), 
    to_connection( "pool_b4", "ccr_b5"), 
    to_connection( "pool_b4", "ccr_b5"), 

    to_UnPool( name='dconv1', offset="(1,0,0)", to="(pool_b5-east)", width=2, height=4, depth=4 ,caption="dconv1"),
    to_UnPool( name='dconv2', offset="(1,0,0)", to="(dconv1-east)", width=2, height=9, depth=9 ,caption="dconv2"),
    to_UnPool( name='dconv3', offset="(1,0,0)", to="(dconv2-east)", width=2, height=16, depth=16 ,caption="dconv3"),
    to_UnPool( name='dconv4', offset="(1,0,0)", to="(dconv3-east)", width=2, height=25, depth=25 ,caption="dconv4"),
    to_UnPool( name='dconv5', offset="(1,0,0)", to="(dconv4-east)", width=2, height=32, depth=32 ,caption="dconv5"),
    to_connection( "pool_b5", "dconv1"), 
    to_connection( "dconv1", "dconv2"), 
    to_connection( "dconv2", "dconv3"), 
    to_connection( "dconv3", "dconv4"), 
    to_connection( "dconv4", "dconv5"), 

    to_skip( "pool_b4", "dconv2", pos=1.5), 
    to_skip( "pool_b3", "dconv3", pos=1.5), 
    to_skip( "pool_b2", "dconv4", pos=1.5), 
    to_skip( "pool_b1", "dconv5", pos=1.5), 

    to_input('../img/TS_mask.jpg' ,to="(35,0,0)", name="output"),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
