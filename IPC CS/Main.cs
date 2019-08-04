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
using System.Timers;
using Emgu.CV;
using Emgu.CV.Structure;

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
        private static System.Timers.Timer timer;

        public Main()
        {
            InitializeComponent();
            Process p = new Process(); // create process (i.e., the python program
            p.StartInfo.FileName = "cmd.exe";
            p.StartInfo.UseShellExecute = false; // make sure we can read the output from stdout
            p.StartInfo.Arguments = "/k cd ../../.. & python IPC.py";
            p.Start();

            timer = new System.Timers.Timer();
            timer.Interval = 40;
            timer.Elapsed += OnTimedEvent;
            timer.AutoReset = true;
            timer.Enabled = false;
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
                                } else if (str == "Alert")
                                {
                                    if (videorecord == 0)
                                    {
                                        videoalert = 1;
                                        recordvideotrigger();
                                    }
                                    lock (_lock)
                                    {
                                        _queue.Enqueue("Received");
                                    }
                                    _signal.Set();
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
            using (var bmpTemp = new Bitmap("display_sharp.jpg"))
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
                    int count = 0;
                    _queue.Enqueue("Start");
                    String text1 = " ";
                    String text2 = " ";
                    String text3 = " ";
                    String text4 = " ";
                    String text5 = " ";
                    String text6 = " ";
                    String text8 = " ";
                    if (textBox1.Text != "")
                    {
                        count++;
                        text1 = textBox1.Text;
                    }
                    if (textBox2.Text != "")
                    {
                        count++;
                        text2 = textBox2.Text;
                    }
                    if (textBox3.Text != "")
                    {
                        count++;
                        text3 = textBox3.Text;
                    }
                    if (textBox4.Text != "")
                    {
                        count++;
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
                    if (textBox8.Text != "")
                    {
                        text8 = textBox8.Text;
                    }
                    if (count == 1)
                    {
                        _queue.Enqueue("1");
                        _queue.Enqueue(text1);
                    }
                    else if (count == 2 || count == 3)
                    {
                        _queue.Enqueue("2");
                        _queue.Enqueue(text1);
                        _queue.Enqueue(text2);
                    }
                    else if (count == 4)
                    {
                        _queue.Enqueue("4");
                        _queue.Enqueue(text1);
                        _queue.Enqueue(text2);
                        _queue.Enqueue(text3);
                        _queue.Enqueue(text4);
                    }
                    else
                    {
                        _queue.Enqueue("0");
                    }
                    _queue.Enqueue(text5);
                    _queue.Enqueue(text6);
                    _queue.Enqueue(text8);
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
            if (textBox7.Text != "")
            {
                lock (_lock)
                {
                    _queue.Enqueue("FaceInput");
                    _queue.Enqueue(textBox7.Text);
                }
                _signal.Set();
            }
        }
        private int alarm = 0;
        private void button4_Click(object sender, EventArgs e)
        {
            if (alarm == 0)
            {
                lock (_lock)
                {
                    _queue.Enqueue("AlarmDeactive");
                }
                button4.Text = "Alarm Activate";
                alarm = 1;
            }
            else
            {
                lock (_lock)
                {
                    _queue.Enqueue("AlarmActive");
                }
                button4.Text = "Alarm Deactivate";
                alarm = 0;
            }
            _signal.Set();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Image img;
            using (var bmpTemp = new Bitmap("display_sharp.jpg"))
            {
                img = new Bitmap(bmpTemp);
            }
            DateTime aDate = DateTime.Now;
            String Datasdate = aDate.ToString("dd MM yyyy HH;mm;ss");
            String photoTime = @"recordimg/"+Datasdate+".jpg";
//            String photoTime = @"D:\skripsi programming\Bedssys\images\recordimg\" + Datasdate + ".jpg";
            try
            {
                img.Save(photoTime, System.Drawing.Imaging.ImageFormat.Jpeg);
            }
            catch
            {

            }
        }
        private int videoalert = 0;
        private int videorecord = 0;
        private VideoWriter writers;
        private int videocounter; 
        private void button3_Click(object sender, EventArgs e)
        {
            recordvideotrigger();
        }
        private void recordvideotrigger()
        {
            if (videorecord == 0)
            {
                if (label10.InvokeRequired)
                {
                    label10.Invoke(new MethodInvoker(delegate { label10.Text = "Stop Video"; }));
                }
                videorecord = 1;
                Image img;
                lock (_lock)
                {
                    using (var bmpTemp = new Bitmap("display_sharp.jpg"))
                    {
                        img = new Bitmap(bmpTemp);
                    }
                }
                DateTime aDate = DateTime.Now;
                String Datasdate = aDate.ToString("dd MM yyyy HH;mm;ss");
                String photoTime = @"recordvideo/" + Datasdate + ".mp4";
                //                String photoTime = @"D:\skripsi programming\Bedssys\images\recordvideo\" + Datasdate + ".mp4";
                int imageHeight = img.Height;
                int imageWidth = img.Width;
                try
                {
                    writers = new VideoWriter(photoTime, VideoWriter.Fourcc('M', 'P', '4', 'V'), 5, new Size(imageWidth, imageHeight), true);
                }
                catch { }
                timer.Enabled = true;
                videocounter = 0;
            }
            else
            {
                if (videocounter > 100 && videoalert == 0)
                {
                    if (label10.InvokeRequired)
                    {
                        label10.Invoke(new MethodInvoker(delegate { label10.Text = "Save Video"; }));
                    }
                    videorecord = 0;
                }
            }
        }
        private void OnTimedEvent(Object source, System.Timers.ElapsedEventArgs e)
        {
            if (videorecord == 1) {

                if (videocounter > 100 && videoalert == 1)
                {
                    if (label10.InvokeRequired)
                    {
                        label10.Invoke(new MethodInvoker(delegate { label10.Text = "Save Video"; }));
                    }
                    videorecord = 0;
                    videoalert = 0;
                    timer.Enabled = false;
                    if (writers != null)
                    {
                        try
                        {
                            writers.Dispose();
                        }
                        catch { }
                    }
                }
                else
                {
                    Bitmap img;
                    using (var bmpTemp = new Bitmap("display_sharp.jpg"))
                    {
                        img = new Bitmap(bmpTemp);
                        Image<Bgr, Byte> imageCV = new Image<Bgr, byte>(img); //Image Class from Emgu.CV
                        Mat mat = imageCV.Mat; //This is your Image converted to Mat
                        writers.Write(mat);
                    }
                    videocounter++;
                }
            }
            else
            {
                timer.Enabled = false;
                if (writers != null)
                {
                    try
                    {
                        writers.Dispose();
                    }
                    catch { }
                }
            }
        }

    }
}