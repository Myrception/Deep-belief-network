using DeepBeliefNeuralNetwork.MLPComponents;
using System;

namespace DeepBeliefNeuralNetwork
{
    /// <summary>
    /// Scheinbar fehlerhaft ?!
    /// </summary>
    internal class XavierInitialization
    {
        private readonly double _stdDev = 1.0;

        public XavierInitialization(int fan_in, int fan_out, IFunktionen AktivierungsfunktionDesLayers)
        {
            if (AktivierungsfunktionDesLayers is MLPComponents.Funktionen.SigmoideFunktion)
            {
                double temp = Math.Sqrt(2.0 / (fan_in + fan_out));
                _stdDev = temp * temp;
            }
            if (AktivierungsfunktionDesLayers is MLPComponents.Funktionen.TangensHyperbolikusFunktion)
            {
                double temp = Math.Pow(2.0 / (fan_in + fan_out), 1.0 / 4.0);
                _stdDev = temp * temp;
            }
            if (AktivierungsfunktionDesLayers is MLPComponents.Funktionen.TanhReLu)
            {
                double temp = Math.Pow(2.0 / (fan_in + fan_out), 1.0 / (Math.Sqrt(2.0)));
                _stdDev = temp * temp;
            }
        }

        /// <summary>
        /// Wahrscheinlichkeitsverteilung wird über die Box-Muller-Methode erstellt
        /// </summary>
        /// <param name="rand"></param>
        /// <returns></returns>
        public double CalculateRandom(Random rand)
        {
            const double mean = 0.0;
            //var Gauss = new MathNet.Numerics.Distributions.Normal(mean,_stdDev,rand);
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + _stdDev * randStdNormal; //random normal(mean,stdDev^2)
            //double randNormal = Gauss.Sample();
            return randNormal;
        }
    }
}