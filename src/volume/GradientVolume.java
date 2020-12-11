/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package volume;

/**
 *
 * @author michel
 */
public class GradientVolume {

    public GradientVolume(Volume vol) {
        volume = vol;
        dimX = vol.getDimX();
        dimY = vol.getDimY();
        dimZ = vol.getDimZ();
        data = new VoxelGradient[dimX * dimY * dimZ];
        compute();
        maxmag = -1.0;
    }

    public VoxelGradient getGradient(int x, int y, int z) {
        return data[x + dimX * (y + dimY * z)];
    }

    public void setGradient(int x, int y, int z, VoxelGradient value) {
        data[x + dimX * (y + dimY * z)] = value;
    }

    public void setVoxel(int i, VoxelGradient value) {
        data[i] = value;
    }

    public VoxelGradient getVoxel(int i) {
        return data[i];
    }

    public int getDimX() {
        return dimX;
    }

    public int getDimY() {
        return dimY;
    }

    public int getDimZ() {
        return dimZ;
    }

    /**
     * Computes the gradient information of the volume according to Levoy's
     * paper.
     */
    private void compute() {
        // TODO 4: Implement gradient computation.
        // this just initializes all gradients to the vector (0,0,0)


        // compute the gradient vector and magnitude of each voxel for 2D transfer function
        // this gradient parameter is used to determine the boundaries between the homogeneous region and transition region in the object
        // first, the gradient difference in the voxel in x-axis, y-axis and z-axis directions is computed, this is the gradient vector
        // then, the gradient magnitude of this gradient vector is computed
        // these gradient vector and magnitude are stored in the respective voxel
        for (int i = 0; i < data.length; i++) {
            data[i] = zero;
        }
        for (int x = 1; x < volume.getDimX() - 1; x++) {
            for (int y = 1; y < volume.getDimY() - 1; y++) {
                for (int z = 1; z < volume.getDimZ() - 1; z++) {
                    float dx = (float) ((volume.getVoxel(x - 1, y, z) - volume.getVoxel(x + 1, y, z)) / 2.0);
                    float dy = (float) ((volume.getVoxel(x, y - 1, z) - volume.getVoxel(x, y + 1, z)) / 2.0);
                    float dz = (float) ((volume.getVoxel(x, y, z - 1) - volume.getVoxel(x, y, z + 1)) / 2.0);
                    
                    // get the value of the VoxelGradient based on the calculated dx, dy, dz
                    VoxelGradient value = new VoxelGradient(dx, dy, dz);
                    
                    //set the VoxelGradient
                    this.setGradient(x, y, z, value);
                    // setGradient(x, y, z, value);
                }
            }
        }        
        
    }

    public double getMaxGradientMagnitude() {
        if (maxmag >= 0) {
            return maxmag;
        } else {
            double magnitude = data[0].mag;
            for (int i = 0; i < data.length; i++) {
                magnitude = data[i].mag > magnitude ? data[i].mag : magnitude;
            }
            maxmag = magnitude;
            return magnitude;
        }
    }

    private int dimX, dimY, dimZ;
    private VoxelGradient zero = new VoxelGradient();
    VoxelGradient[] data;
    Volume volume;
    double maxmag;
}
