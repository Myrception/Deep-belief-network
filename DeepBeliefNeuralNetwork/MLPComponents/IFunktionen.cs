using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents
{
    public interface IFunktionen
    {
        double BerechneWert(double input, double alpha);

        double BerechneAbleitung(double input, double alpha);

        double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers);
    }
}