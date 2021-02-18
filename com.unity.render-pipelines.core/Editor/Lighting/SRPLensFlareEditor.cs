using System.IO;
using UnityEditor;
using UnityEngine;

namespace UnityEditor.Rendering
{
    /// <summary>
    /// SRPLensFlareEditor shows how the SRP Lens Flare Asset is shown in the UI
    /// </summary>
    [CustomEditor(typeof(SRPLensFlareData))]
    [HelpURL(UnityEngine.Rendering.Documentation.baseURL + UnityEngine.Rendering.Documentation.version + UnityEngine.Rendering.Documentation.subURL + "Common/srp-lens-flare-asset" + UnityEngine.Rendering.Documentation.endURL)]
    public class SRPLensFlareEditor : Editor
    {
        SerializedProperty m_Intensity;
        SerializedProperty m_ScaleCurve;
        SerializedProperty m_PositionCurve;
        SerializedProperty m_Elements;

        /// <summary>
        /// Prepare the code for the UI
        /// </summary>
        public void OnEnable()
        {
            PropertyFetcher<SRPLensFlareData> entryPoint = new PropertyFetcher<SRPLensFlareData>(serializedObject);
            m_Intensity = entryPoint.Find(x => x.globalIntensity);
            m_ScaleCurve = entryPoint.Find(x => x.scaleCurve);
            m_PositionCurve = entryPoint.Find(x => x.positionCurve);
            m_Elements = entryPoint.Find(x => x.elements);
        }

        /// <summary>
        /// Implement this function to make a custom inspector
        /// </summary>
        public override void OnInspectorGUI()
        {
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_Intensity, Styles.intensity);
            if (EditorGUI.EndChangeCheck())
            {
                m_Intensity.serializedObject.ApplyModifiedProperties();
            }
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_ScaleCurve, Styles.scaleCurve);
            EditorGUILayout.PropertyField(m_PositionCurve, Styles.positionCurve);
            if (EditorGUI.EndChangeCheck())
            {
                m_ScaleCurve.serializedObject.ApplyModifiedProperties();
                m_PositionCurve.serializedObject.ApplyModifiedProperties();
            }
            EditorGUI.BeginChangeCheck();
            SRPLensFlareData lensFlareDat = m_Elements.serializedObject.targetObject as SRPLensFlareData;
            int countBefore = lensFlareDat != null && lensFlareDat.elements != null ? lensFlareDat.elements.Length : 0;
            EditorGUILayout.PropertyField(m_Elements, Styles.elements);
            if (EditorGUI.EndChangeCheck())
            {
                m_Elements.serializedObject.ApplyModifiedProperties();
                int countAfter = lensFlareDat != null && lensFlareDat.elements != null ? lensFlareDat.elements.Length : 0;
                if (countAfter > countBefore)
                {
                    for (int i = countBefore; i < countAfter; ++i)
                    {
                        lensFlareDat.elements[i] = new SRPLensFlareDataElement(); // Set Default values
                    }
                    m_Elements.serializedObject.Update();
                }
            }
        }

        sealed class Styles
        {
            static public readonly GUIContent intensity = new GUIContent("Intensity", "Modulate the whole lens flare.");
            static public readonly GUIContent scaleCurve = new GUIContent("Scale Curve", "Curve between 0 and 1 which describes the scale of each element, if the relative position is negative HDRP will read the negative part of the curve, the positive part otherwise.");
            static public readonly GUIContent positionCurve = new GUIContent("Position Curve", "Curve between -1 and 1 which describes the scale of each element, if the relative position is negative HDRP will read the negative part of the curve, the positive part otherwise.");
            static public readonly GUIContent elements = new GUIContent("Elements", "List of elements in the Lens Flare.");
        }
    }
}
