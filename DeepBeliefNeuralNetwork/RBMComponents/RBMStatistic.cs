using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.RBMComponents
{
    public class RBMStatistic
    {
        public double positivStatistic(RBMNeurons visibleUnit, RBMNeurons HiddenUnit)
        {
            return visibleUnit.Probability * HiddenUnit.Probability;
        }
    }
}
