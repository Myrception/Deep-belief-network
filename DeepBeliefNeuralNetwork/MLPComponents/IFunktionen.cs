using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.MLPComponents
{
    public interface IFunktionen
    {
        double BerechneWert(double input, double alpha);

        double BerechneAbleitung(double input, double alpha);
        double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers);
    }
}
