namespace Loupedeck.ExamplePlugin
{
    using System;
    using System.Net.Http;
    using System.Threading.Tasks;

    internal static class DocTrackClient
    {
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };

        public static void Send(String route)
        {
            Task.Run(async () =>
            {
                try
                {
                    await Http.PostAsync($"http://127.0.0.1:8765/{route}", null);
                }
                catch (Exception ex)
                {
                    PluginLog.Error(ex, $"DocTrack '{route}' failed (is the Python app running?)");
                }
            });
        }
    }

    internal static class Icons
    {
        public static BitmapImage Get(String name, PluginImageSize size)
        {
            var img = PluginResources.ReadImage(PluginResources.FindFile($"{name}.png"));

            using (var b = new BitmapBuilder(size))
            {
                b.Clear(BitmapColor.Black);
                b.DrawImage(img, 0, 0, b.Width, b.Height);
                return b.ToImage();
            }
        }
    }

    public class FlipCommand : PluginDynamicCommand
    {
        public FlipCommand()
            : base(displayName: "Mirror scan", description: "Mirror the scan", groupName: "DocTrack")
        {
        }

        protected override void RunCommand(String actionParameter) => DocTrackClient.Send("mirror");

        protected override BitmapImage GetCommandImage(String actionParameter, PluginImageSize imageSize)
            => Icons.Get("mirror", imageSize);
    }

    public class ReleaseCommand : PluginDynamicCommand
    {
        public ReleaseCommand()
            : base(displayName: "Release", description: "Release the locked document", groupName: "DocTrack")
        {
        }

        protected override void RunCommand(String actionParameter) => DocTrackClient.Send("release");

        protected override BitmapImage GetCommandImage(String actionParameter, PluginImageSize imageSize)
            => Icons.Get("release", imageSize);
    }
}

namespace Loupedeck.ExamplePlugin
{
    using System;
    using System.Collections.Generic;

    public class FormatFolder : PluginDynamicFolder
    {
        private static readonly (String Route, String Label)[] Formats =
        {
            ("auto",  "Auto"),
            ("a4p",   "A4 portrait"),
            ("a4l",   "A4 landscape"),
            ("photo", "20x25 (4:5)"),
        };

        private String _current = "auto";   // matches the Python default

        public FormatFolder()
        {
            this.DisplayName = "Format";
            this.GroupName = "DocTrack";
            this.Navigation = PluginDynamicFolderNavigation.ButtonArea;
        }

        // The buttons shown on the folder page: a Back button, then one per format.
        public override IEnumerable<String> GetButtonPressActionNames(DeviceType deviceType)
        {
            var names = new List<String> { PluginDynamicFolder.NavigateUpActionName };
            foreach (var f in Formats)
            {
                names.Add(this.CreateCommandName(f.Route));
            }
            return names;
        }

        public override void RunCommand(String actionParameter)
        {
            this._current = actionParameter;
            DocTrackClient.Send($"format/{actionParameter}");

            // refresh the icons so the active one shows the check mark
            foreach (var f in Formats)
            {
                this.CommandImageChanged(f.Route);
            }
        }

        // Icon of the folder button itself
        public override BitmapImage GetButtonImage(PluginImageSize imageSize)
            => Icons.Get("format", imageSize);

        // Icons of the buttons inside the folder, with a check mark on the active one
        public override BitmapImage GetCommandImage(String actionParameter, PluginImageSize imageSize)
        {
            using (var b = new BitmapBuilder(imageSize))
            {
                b.DrawImage(Icons.Get(actionParameter, imageSize), 0, 0, b.Width, b.Height);

                if (actionParameter == this._current)
                {
                    b.DrawText("✓", b.Width - 26, 2, 24, 24, new BitmapColor(64, 196, 255), 20);
                }
                return b.ToImage();
            }
        }

        // Label is empty because the icon carries the meaning
        public override String GetCommandDisplayName(String actionParameter, PluginImageSize imageSize) => "";
    }
}

namespace Loupedeck.ExamplePlugin
{
    using System;

    public class DocTrackDialAdjustment : PluginDynamicAdjustment
    {
        private Int32 _quarterTurns = 0;
        private Boolean _mirrored = false;

        // hasReset: true makes the dial press call RunCommand below.
        public DocTrackDialAdjustment()
            : base(displayName: "Rotate", description: "Turn to rotate 90°, press to mirror", groupName: "DocTrack", hasReset: true)
        {
        }

        protected override void ApplyAdjustment(String actionParameter, Int32 diff)
        {
            // The Python app only rotates one way, so a counter-clockwise tick is 3 clockwise rotations.
            var times = diff > 0 ? diff : -diff * 3;
            for (var i = 0; i < times; i++)
            {
                DocTrackClient.Send("rotate");
            }

            this._quarterTurns = ((this._quarterTurns + diff) % 4 + 4) % 4;
            this.AdjustmentValueChanged();
        }

        protected override void RunCommand(String actionParameter)
        {
            DocTrackClient.Send("mirror");
            this._mirrored = !this._mirrored;
            this.AdjustmentValueChanged();
        }

        protected override String GetAdjustmentValue(String actionParameter) =>
            $"{this._quarterTurns * 90}°{(this._mirrored ? " M" : "")}";

        // Icon shown on the dial / button this adjustment is assigned to
        protected override BitmapImage GetAdjustmentImage(String actionParameter, PluginImageSize imageSize)
            => Icons.Get("rotate", imageSize);
    }
}

namespace Loupedeck.ExamplePlugin
{
    using System;

    // Display the speaker's face in a square.
    public class SpeakerFaceCommand : PluginDynamicCommand
    {
        public SpeakerFaceCommand()
            : base(displayName: "Face", description: "Display the speaker's face in a square", groupName: "DocTrack")
        {
        }

        protected override void RunCommand(String actionParameter) => DocTrackClient.Send("face");

        protected override BitmapImage GetCommandImage(String actionParameter, PluginImageSize imageSize)
            => Icons.Get("speaker_face", imageSize);
    }

    // Share the PDF.
    public class SharePdfCommand : PluginDynamicCommand
    {
        public SharePdfCommand()
            : base(displayName: "Share PDF", description: "Share the PDF", groupName: "DocTrack")
        {
        }

        protected override void RunCommand(String actionParameter) => DocTrackClient.Send("share/pdf");

        protected override BitmapImage GetCommandImage(String actionParameter, PluginImageSize imageSize)
            => Icons.Get("share_pdf", imageSize);
    }

    // Share the PDF together with the notes.
    public class SharePdfNotesCommand : PluginDynamicCommand
    {
        public SharePdfNotesCommand()
            : base(displayName: "Share PDF + notes", description: "Share the PDF with notes", groupName: "DocTrack")
        {
        }

        protected override void RunCommand(String actionParameter) => DocTrackClient.Send("share/pdf-notes");

        protected override BitmapImage GetCommandImage(String actionParameter, PluginImageSize imageSize)
            => Icons.Get("share_pdf_notes", imageSize);
    }

    // Audio recording on/off. The icon shows the current state.
    public class AudioToggleCommand : PluginDynamicCommand
    {
        private Boolean _on = false;  // not recording at start; adjust once the Python side reports its state

        public AudioToggleCommand()
            : base(displayName: "Audio on/off", description: "Turn audio recording on or off", groupName: "DocTrack")
        {
        }

        protected override void RunCommand(String actionParameter)
        {
            this._on = !this._on;
            DocTrackClient.Send(this._on ? "audio/on" : "audio/off");
            this.ActionImageChanged();
        }

        protected override BitmapImage GetCommandImage(String actionParameter, PluginImageSize imageSize)
            => Icons.Get(this._on ? "mic_on" : "mic_off", imageSize);
    }

    // Stop: asks the running Python app to quit (it closes the windows and releases the camera).
    public class StopCommand : PluginDynamicCommand
    {
        public StopCommand()
            : base(displayName: "Stop", description: "Stop DocTrack", groupName: "DocTrack")
        {
        }

        protected override void RunCommand(String actionParameter) => DocTrackClient.Send("quit");

        protected override BitmapImage GetCommandImage(String actionParameter, PluginImageSize imageSize)
            => Icons.Get("stop", imageSize);
    }
}