using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TestSO
{

    public partial class Main : Form
    {

        private readonly object _lock = new object();
        private readonly Queue<string> _queue = new Queue<string>();
        private readonly AutoResetEvent _signal = new AutoResetEvent(false);
        private static NamedPipeServerStream server;
        private BinaryReader br;
        private BinaryWriter bw;
        public Main()
        {
            InitializeComponent();
            server = new NamedPipeServerStream("testing");
            br = new BinaryReader(server);
            bw = new BinaryWriter(server);
            new Thread(new ThreadStart(ProducerThread)).Start();
        }
        private void Main_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (server != null)
            {
                server.Close();
                server.Dispose();
                server = null;
            }
            Application.Exit();
        }

        private String mode = "normal";
        void ProducerThread()
        {
            while (true)
            {
                _signal.WaitOne();
                string item = string.Empty;
                do
                {
                    item = string.Empty;
                    lock (_lock)
                    {
                        if (_queue.Count > 0)
                        {
                            item = _queue.Dequeue();
                        }
                    }

                    if (item != string.Empty)
                    {

                        try
                        {

                            if (server != null && !server.IsConnected)
                                server.WaitForConnection();

                            if (server != null && server.IsConnected)
                            {
                                var str = new string(item.ToString().ToArray());

                                var buf = Encoding.ASCII.GetBytes(str);
                                bw.Write((uint)buf.Length);
                                bw.Write(buf);
                            }
                            if (server != null && server.IsConnected)
                            {
                                var len = (int)br.ReadUInt32();
                                var str = new string(br.ReadChars(len));
                                if (str == "Image")
                                {
                                    searchimage();
                                } else if (str == "Wait")
                                {
                                    lock (_lock)
                                    {
                                        while (_queue.Count > 0)
                                        {
                                            String throwingaway = _queue.Dequeue();
                                        }
                                    }
                                    _signal.Set();
                                } 
                            }
                        }
                        catch (Exception EX)
                        {
                            MessageBox.Show(EX.Message.ToString());
                        }
                    }
                }
                while (item != string.Empty);
            }
        }
        private void searchimage()
        {
            Image img;
            using (var bmpTemp = new Bitmap("D:\\skripsi programming\\Bedssys\\display_sharp.jpg"))
            {
                img = new Bitmap(bmpTemp);
            }
            //load image in picturebox
            pictureBox1.Image = img;
            lock (_lock)
            {
                _queue.Enqueue("Received");
            }
            _signal.Set();
        }
        private int checkbutton = 0;
        private void button1_Click(object sender, EventArgs e)
        {
            if (checkbutton == 0) {
                lock (_lock)
                {
                    _queue.Enqueue("Start");
                    String text1 = " ";
                    String text2 = " ";
                    String text3 = " ";
                    String text4 = " ";
                    String text5 = " ";
                    String text6 = " ";
                    if (textBox1.Text != "")
                    {
                        text1 = textBox1.Text;
                    }
                    if (textBox2.Text != "")
                    {
                        text2 = textBox2.Text;
                    }
                    if (textBox3.Text != "")
                    {
                        text3 = textBox3.Text;
                    }
                    if (textBox4.Text != "")
                    {
                        text4 = textBox4.Text;
                    }
                    if (textBox5.Text != "")
                    {
                        text5 = textBox5.Text;
                    }
                    if (textBox6.Text != "")
                    {
                        text6 = textBox6.Text;
                    }
                    _queue.Enqueue(text1);
                    _queue.Enqueue(text2);
                    _queue.Enqueue(text3);
                    _queue.Enqueue(text4);
                    _queue.Enqueue(text5);
                    _queue.Enqueue(text6);
                    _queue.Enqueue("Received");
                }
                button1.Text = "Stop Kamera";
                checkbutton = 1;
            }
            else
            {
                lock (_lock)
                {
                    _queue.Enqueue("Stop");
                }
                button1.Text = "Mulai Kamera";
                checkbutton = 0;
            }
            _signal.Set();
        }

        private void button5_Click(object sender, EventArgs e)
        {
            if(mode == "normal") {
                lock (_lock)
                {
                    _queue.Enqueue("Normal");
                }
                button5.Text = "Recognition Mode";
                mode = "recognition";
            }
            else
            {
                lock (_lock)
                {
                    _queue.Enqueue("Recognition");
                }
                button5.Text = "Normal Mode";
                mode = "normal";
            }
            _signal.Set();
        }

        private void button6_Click(object sender, EventArgs e)
        {
            lock (_lock)
            {
                _queue.Enqueue("FaceInput");
                _queue.Enqueue("FaceName");
            }
            _signal.Set();
        }
        private int alarm;
        private void button4_Click(object sender, EventArgs e)
        {

            if (alarm == 0)
            {
                lock (_lock)
                {
                    _queue.Enqueue("AlarmActive");
                }
                button4.Text = "Alarm Activate";
                alarm = 1;
            }
            else
            {
                lock (_lock)
                {
                    _queue.Enqueue("AlarmDeactive");
                }
                button4.Text = "Alarm Deactivate";
                alarm = 0;
            }
            _signal.Set();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            lock (_lock)
            {
                _queue.Enqueue("SaveImage");
            }
            _signal.Set();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            lock (_lock)
            {
                _queue.Enqueue("SaveVideo");
            }
            _signal.Set();
        }
    }
}