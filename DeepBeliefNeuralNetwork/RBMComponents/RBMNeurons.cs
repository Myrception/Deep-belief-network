using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.RBMComponents
{
    public class RBMNeurons
    {
        public int Index { get; set; }
        public double BinaryState { get; set; }
        public double Probability { get; set; }
    }
}
