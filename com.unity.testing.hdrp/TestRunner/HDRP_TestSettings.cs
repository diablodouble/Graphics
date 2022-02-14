using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using NUnit.Framework;
using UnityEditor;
using UnityEngine.TestTools;
using UnityEngine.TestTools.Graphics;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Events;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using UnityEngine.SceneManagement;

[ExecuteAlways]
public class HDRP_TestSettings : GraphicsTestSettings
{
    public UnityEngine.Events.UnityEvent doBeforeTest;
    public int captureFramerate = 0;
    public int waitFrames = 0;
    public bool xrCompatible = true;

    [UnityEngine.Range(1.0f, 10.0f)]
    public float xrThresholdMultiplier = 1.0f;

    public bool checkMemoryAllocation = true;

    public RenderPipelineAsset renderPipelineAsset;

    [Tooltip("RP Asset change is only effective after a frame is render")]
    public bool forceCameraRenderDuringSetup = false;

    void Awake()
    {
        if (renderPipelineAsset == null)
        {
            Debug.LogWarning("No RenderPipelineAsset has been assigned in the test settings. This may result in a wrong test.");
            return;
        }

        var currentRP = GraphicsSettings.renderPipelineAsset;

        if (currentRP != renderPipelineAsset)
        {
            quitDebug.AppendLine($"{SceneManager.GetActiveScene().name} RP asset change: {((currentRP == null) ? "null" : currentRP.name)} => {renderPipelineAsset.name}");

            GraphicsSettings.renderPipelineAsset = renderPipelineAsset;

            // Render pipeline is only reconstructed when a frame is renderer
            // If scene requires lightmap baking, we have to force it
            // Currently Camera.Render() fails on mac so we have to filter out the tests that rely on forceCameraRenderDuringSetup (like 2120 for APV).
            // But since setup is run regardless of the filter we add this explicit check on platform
            if (forceCameraRenderDuringSetup && !Application.isPlaying && Application.platform != RuntimePlatform.OSXEditor)
                Camera.main.Render();
        }
    }

    static StringBuilder quitDebug = new StringBuilder();

    void OnApplicationQuit()
    {
        if (quitDebug.Length == 0) return;

        Debug.Log($"Scenes that needed to change the RP asset:{Environment.NewLine}{quitDebug.ToString()}");

        quitDebug.Clear();
    }
}
