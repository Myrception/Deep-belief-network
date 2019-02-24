using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    public class Softmax : IFunktionen
    {
        public double BerechneWert(double input, double alpha)
        {
            return (Math.Exp(input));
        }

        public double BerechneAbleitung(double input, double alpha)
        {
            throw new NotImplementedException();
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layer)
        {
            double sumOfAlNetInput = 0, output = 0;
            foreach (var mlpNeuron in layer[layer.Count - 1])
            {
                sumOfAlNetInput += Math.Exp(mlpNeuron.NetInput);
            }
            output = Math.Exp(input);
            //output += (Math.Exp(input) / sumOfAlNetInput) * (1 - (Math.Exp(input) / sumOfAlNetInput));
            //foreach (var mlpNeuron in layer[layer.Count-1])
            //{
            //    if (mlpNeuron.NetInput > input || mlpNeuron.NetInput < input)
            //    {
            //        output += -((Math.Exp(input) / sumOfAlNetInput) * (Math.Exp(mlpNeuron.NetInput) / sumOfAlNetInput));
            //    }
            //}
            //return output;
            return (output / sumOfAlNetInput) * (1 - (output / sumOfAlNetInput));
            //return alpha * (1 / (1 + Math.Pow(Math.E, -input))) * (1 - (1 / (1 + Math.Pow(Math.E, -input))));
        }
    }
}