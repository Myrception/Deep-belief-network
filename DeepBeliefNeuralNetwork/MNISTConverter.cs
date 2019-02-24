//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;

//namespace DeepBeliefNeuralNetwork
//{
//    class MNISTConverter
//    {
//        string fileName = @"C:\Users\Joseph\Desktop\ProjektKI&S\Muster\7_7_7_logistic_025TrainingsMuster.csv";
//        StreamReader reader = new StreamReader(fileName);
//        string line;
//        string[] splitted;
//        char[] separator = ";".ToCharArray();
//        List<TrainingsMuster> patternFromFile = new List<TrainingsMuster>();

//        line = reader.ReadLine();
//            splitted = line.Split(separator);
//            foreach (var item in splitted)
//            {
//                line = item.Substring(0, 7);
//                item.ToLower();
//                if (line.ToString() == "eingabe")
//                {
//                    anzahleingabe++;
//                }
//                if (line.ToString() == "ausgabe")
//                {
//                    anzahlausgabe++;
//                }
//            }
//            while (!reader.EndOfStream)
//            {
//                line = reader.ReadLine();
//                splitted = line.Split(separator);
//                TrainingsMuster muster = new TrainingsMuster();
//muster.Eingabevektor = new double[anzahleingabe];
//                muster.Targetvektor = new double[anzahlausgabe];
//                for (int i = 0; i<splitted.Length; i++)
//                {
//                    if (i<anzahleingabe)
//                    {
//                        muster.Eingabevektor[i] = Convert.ToDouble(splitted[i]);
//                    }
//                    else
//                    {
//                        muster.Targetvektor[i - anzahleingabe] = Convert.ToDouble(splitted[i]);
//                    }
//                }
//                patternFromFile.Add(muster);
//            }
//    }
//}