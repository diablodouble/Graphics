#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Packing.hlsl"
#include "Packages/com.unity.render-pipelines.high-definition/Runtime/Material/DiffusionProfile/DiffusionProfileSettings.cs.hlsl"
#include "Packages/com.unity.render-pipelines.high-definition/Runtime/Material/DiffusionProfile/DiffusionProfile.hlsl"

// ----------------------------------------------------------------------------
// helper functions
// ----------------------------------------------------------------------------

// 0: [ albedo = albedo ]
// 1: [ albedo = 1 ]
// 2: [ albedo = sqrt(albedo) ]
uint GetSubsurfaceScatteringTexturingMode(int diffusionProfile)
{
    uint texturingMode = 0;

#if defined(SHADERPASS) && (SHADERPASS == SHADERPASS_SUBSURFACE_SCATTERING)
    // If the SSS pass is executed, we know we have SSS enabled.
    bool enableSss = true;
    // SSS in HDRP is a screen space effect thus, it is not available for the lighting-based ray tracing passes (RTR, RTGI and RR). Thus we need to disable
    // the feature if we are in a ray tracing pass.
#elif defined(SHADERPASS) && ((SHADERPASS == SHADERPASS_RAYTRACING_INDIRECT) || (SHADERPASS == SHADERPASS_RAYTRACING_FORWARD))
    // If the SSS pass is executed, we know we have SSS enabled.
    bool enableSss = false;
#elif SHADEROPTIONS_QUALITY_LITE
    // For HDRP Lite, SSS is not done in screenspace
    bool enableSss = false;
#else
    bool enableSss = _EnableSubsurfaceScattering != 0;
#endif

    if (enableSss)
    {
        bool performPostScatterTexturing = IsBitSet(_TexturingModeFlags, diffusionProfile);

        if (performPostScatterTexturing)
        {
            // Post-scatter texturing mode: the albedo is only applied during the SSS pass.
        #if defined(SHADERPASS) && (SHADERPASS != SHADERPASS_SUBSURFACE_SCATTERING)
            texturingMode = 1;
        #endif
        }
        else
        {
            // Pre- and post- scatter texturing mode.
            texturingMode = 2;
        }
    }

    return texturingMode;
}

// Returns the modified albedo (diffuse color) for materials with subsurface scattering.
// See GetSubsurfaceScatteringTexturingMode() above for more details.
// Ref: Advanced Techniques for Realistic Real-Time Skin Rendering.
float3 ApplySubsurfaceScatteringTexturingMode(uint texturingMode, float3 color)
{
    switch (texturingMode)
    {
        case 2:  color = sqrt(color); break;
        case 1:  color = 1;           break;
        default: color = color;       break;
    }

    return color;
}

// ----------------------------------------------------------------------------
// Encoding/decoding SSS buffer functions
// ----------------------------------------------------------------------------

struct SSSData
{
    float3 diffuseColor;
    float  subsurfaceMask;
    uint   diffusionProfileIndex;
};

#define SSSBufferType0 float4 // Must match GBufferType0 in deferred

// SSSBuffer texture declaration
TEXTURE2D_X(_SSSBufferTexture);

// Note: The SSS buffer used here is sRGB
void EncodeIntoSSSBuffer(SSSData sssData, uint2 positionSS, out SSSBufferType0 outSSSBuffer0)
{
    outSSSBuffer0 = float4(sssData.diffuseColor, PackFloatInt8bit(sssData.subsurfaceMask, sssData.diffusionProfileIndex, 16));
}

// Note: The SSS buffer used here is sRGB
void DecodeFromSSSBuffer(float4 sssBuffer, uint2 positionSS, out SSSData sssData)
{
    sssData.diffuseColor = sssBuffer.rgb;
    UnpackFloatInt8bit(sssBuffer.a, 16, sssData.subsurfaceMask, sssData.diffusionProfileIndex);
}

void DecodeFromSSSBuffer(uint2 positionSS, out SSSData sssData)
{
    float4 sssBuffer = LOAD_TEXTURE2D_X(_SSSBufferTexture, positionSS);
    DecodeFromSSSBuffer(sssBuffer, positionSS, sssData);
}

#define OUTPUT_SSSBUFFER(NAME) out SSSBufferType0 MERGE_NAME(NAME, 0)

#define ENCODE_INTO_SSSBUFFER(SURFACE_DATA, UNPOSITIONSS, NAME) EncodeIntoSSSBuffer(ConvertSurfaceDataToSSSData(SURFACE_DATA), UNPOSITIONSS, MERGE_NAME(NAME, 0))

#define DECODE_FROM_SSSBUFFER(UNPOSITIONSS, SSS_DATA) DecodeFromSSSBuffer(UNPOSITIONSS, SSS_DATA)

// In order to support subsurface scattering, we need to know which pixels have an SSS material.
// It can be accomplished by reading the stencil buffer.
// A faster solution (which avoids an extra texture fetch) is to simply make sure that
// all pixels which belong to an SSS material are not black (those that don't always are).
// We choose the blue color channel since it's perceptually the least noticeable.
float3 TagLightingForSSS(float3 subsurfaceLighting)
{
    subsurfaceLighting.b = max(subsurfaceLighting.b, HALF_MIN);
    return subsurfaceLighting;
}

// See TagLightingForSSS() for details.
bool TestLightingForSSS(float3 subsurfaceLighting)
{
    return subsurfaceLighting.b > 0;
}

// ----------------------------------------------------------------------------
// Helper functions to use SSS/Transmission with a material
// ----------------------------------------------------------------------------

// Following function allow to easily setup SSS and transmission inside a material.
// User can request either SSS functions, or Transmission functions, or both, by defining MATERIAL_INCLUDE_SUBSURFACESCATTERING and/or MATERIAL_INCLUDE_TRANSMISSION
// before including this file.
// + It require that the material follow naming convention for properties inside BSDFData

// struct BSDFData
// {
//     (...)
//     // Share for SSS and Transmission
//     uint materialFeatures;
//     uint diffusionProfile;
//     // For SSS
//     float3 diffuseColor;
//     float3 fresnel0;
//     float subsurfaceMask;
//     // For transmission
//     float thickness;
//     bool useThickObjectMode;
//     float3 transmittance;
//     perceptualRoughness; // Only if user chose to support DisneyDiffuse
//     (...)
// }

// Note: Transmission functions for light evaluation are also included in LightEvaluation.hlsl file based on the MATERIAL_INCLUDE_TRANSMISSION
#define MATERIALFEATUREFLAGS_SSS_TRANSMISSION_START (1 << 16) // It should be safe to start these flags

#define MATERIALFEATUREFLAGS_SSS_OUTPUT_SPLIT_LIGHTING         ((MATERIALFEATUREFLAGS_SSS_TRANSMISSION_START) << 0)
#define MATERIALFEATUREFLAGS_SSS_TEXTURING_MODE_OFFSET FastLog2((MATERIALFEATUREFLAGS_SSS_TRANSMISSION_START) << 1) // Note: The texture mode is 2bit, thus go from '<< 1' to '<< 3'
// Flags used as a shortcut to know if we have thick mode transmission
// It is important to keep this flag pointing at the inverse of the current diffusion profile thickness mode, i.e. the
// current diffusion profile thickness mode is thin because we don't want to sample shadows for the default profile
// so this define is set to thick mode. It is important to keep it as is because when we initialize the BSDF datas
// we assume that all neutral values including the thickness mode are 0 (so by default when we shade a material that
// doesn't have transmission on a tile with the material feature transmission enabled, we don't evaluate the diffusion
// profile because the thick flag is not set (for pixels that have transmission, we force the flags in a per-pixel
// material feature)).
#define MATERIALFEATUREFLAGS_TRANSMISSION_MODE_THICK_OBJECT     ((MATERIALFEATUREFLAGS_SSS_TRANSMISSION_START) << 3)

// 15 degrees
#define TRANSMISSION_WRAP_ANGLE (PI/12)
#define TRANSMISSION_WRAP_LIGHT cos(PI/2 - TRANSMISSION_WRAP_ANGLE)

#ifdef MATERIAL_INCLUDE_SUBSURFACESCATTERING

void FillMaterialSSS(uint diffusionProfileIndex, float subsurfaceMask, inout BSDFData bsdfData)
{
    bsdfData.diffusionProfileIndex = diffusionProfileIndex;
    bsdfData.fresnel0 = _TransmissionTintsAndFresnel0[diffusionProfileIndex].a;
    bsdfData.shapeParam = _ShapeParamsAndMaxScatterDists[diffusionProfileIndex].rgb;
    bsdfData.worldScale = _WorldScalesAndFilterRadiiAndThicknessRemaps[diffusionProfileIndex].r;
    bsdfData.filterRadius = _WorldScalesAndFilterRadiiAndThicknessRemaps[diffusionProfileIndex].g;
    bsdfData.subsurfaceMask = subsurfaceMask;
#if !SHADEROPTIONS_QUALITY_LITE
    bsdfData.materialFeatures |= MATERIALFEATUREFLAGS_SSS_OUTPUT_SPLIT_LIGHTING;
#endif
    bsdfData.materialFeatures |= GetSubsurfaceScatteringTexturingMode(diffusionProfileIndex) << MATERIALFEATUREFLAGS_SSS_TEXTURING_MODE_OFFSET;
}

bool ShouldOutputSplitLighting(BSDFData bsdfData)
{
    return HasFlag(bsdfData.materialFeatures, MATERIALFEATUREFLAGS_SSS_OUTPUT_SPLIT_LIGHTING);
}

float3 GetModifiedDiffuseColorForSSS(BSDFData bsdfData)
{
    // Subsurface scattering mode
    uint   texturingMode = (bsdfData.materialFeatures >> MATERIALFEATUREFLAGS_SSS_TEXTURING_MODE_OFFSET) & 3;
    return ApplySubsurfaceScatteringTexturingMode(texturingMode, bsdfData.diffuseColor);
}

#endif

#ifdef MATERIAL_INCLUDE_TRANSMISSION

// Assume that bsdfData.diffusionProfileIndex is init
void FillMaterialTransmission(uint diffusionProfileIndex, float thickness, inout BSDFData bsdfData)
{
    float2 remap = _WorldScalesAndFilterRadiiAndThicknessRemaps[diffusionProfileIndex].zw;

    bsdfData.diffusionProfileIndex = diffusionProfileIndex;
    bsdfData.fresnel0              = _TransmissionTintsAndFresnel0[diffusionProfileIndex].a;
    bsdfData.thickness             = remap.x + remap.y * thickness;

    // The difference between the thin and the regular (a.k.a. auto-thickness) modes is the following:
    // * in the thin object mode, we assume that the geometry is thin enough for us to safely share
    // the shadowing information between the front and the back faces;
    // * the thin mode uses baked (textured) thickness for all transmission calculations;
    // * the thin mode uses wrapped diffuse lighting for the NdotL;
    // * the auto-thickness mode uses the baked (textured) thickness to compute transmission from
    // indirect lighting and non-shadow-casting lights; for shadowed lights, it calculates
    // the thickness using the distance to the closest occluder sampled from the shadow map.
    // If the distance is large, it may indicate that the closest occluder is not the back face of
    // the current object. That's not a problem, since large thickness will result in low intensity.
    bool useThickObjectMode = !IsBitSet(asuint(_TransmissionFlags), diffusionProfileIndex);

    bsdfData.materialFeatures |= useThickObjectMode ? MATERIALFEATUREFLAGS_TRANSMISSION_MODE_THICK_OBJECT : 0;

    // Compute transmittance using baked thickness here. It may be overridden for direct lighting
    // in the auto-thickness mode (but is always used for indirect lighting).
    bsdfData.transmittance = ComputeTransmittanceDisney(_ShapeParamsAndMaxScatterDists[diffusionProfileIndex].rgb,
                                                        _TransmissionTintsAndFresnel0[diffusionProfileIndex].rgb,
                                                        bsdfData.thickness);
}

#endif

#if defined(MATERIAL_INCLUDE_SUBSURFACESCATTERING) || defined(MATERIAL_INCLUDE_TRANSMISSION)

uint FindDiffusionProfileIndex(uint diffusionProfileHash)
{
    if (diffusionProfileHash == 0)
        return 0;

    uint diffusionProfileIndex = 0;
    uint i = 0;

    // Fetch the 4 bit index number by looking for the diffusion profile unique ID:
    for (i = 0; i < _DiffusionProfileCount; i++)
    {
        if (_DiffusionProfileHashTable[i].x == diffusionProfileHash)
        {
            diffusionProfileIndex = i;
            break;
        }
    }

    return diffusionProfileIndex;
}

#endif

float3 IntegrateDiffuseScattering(float3 NdotL, float radius, float3 shapeParam, int steps)
{
	float limit = PI * 0.5f;
	float inc = 2.0f * limit / (float)steps;

    float3 theta = acos(NdotL);
	float3 totalWeights = EvalBurleyDiffusionProfile(0, shapeParam);
	float3 totalLight = totalWeights * saturate(cos(theta));
	for (float x = -limit; x <= limit; x += inc)
	{
		float3 diffuse = saturate(cos(theta + x));
		float dist = abs(2.0f * radius * sin(x * 0.5f));
		float3 weights = EvalBurleyDiffusionProfile(dist, shapeParam);
		//float3 weights = EvalGaussianDiffusionProfile(dist);

		totalWeights += weights;
		totalLight += diffuse * weights;
	}
	return saturate(totalLight / totalWeights);
}

float IntegrateDiffuseScattering(float NdotL, float radius, float shapeParam)
{
	float x = 0.5f + 0.5f * NdotL;
	float y = 1.0f - (radius - 1.0f) / 15.0f;
	float s = 1.0f / shapeParam;

	float x2 = x * x;
	float fit_a = x * (0.297809419735113f * x + 0.409123767595839f); // madd mul
	float fit_b_1 = x2 * (2.55383104671677f * x2 - 0.479377197902363f) + 0.0891580536081126f * x; // madd mul madd
	float fit_b_2 = x * 1.895f - 0.925f; // madd

	float fit_b = x < 0.6339f ? fit_b_1 : fit_b_2;

	float lambda_scat = s < 0.07748f ? 24 * s*s : 0.29f * (1 - 0.925 / (s + 0.17)) + 0.938;
	float fit_min_scat_a = x < 0.55 ? saturate((x - 0.4)* (x - 0.4)* (x - 0.4) * 30) : 2 * x - 1;
	float fit_min_scat_b = saturate(2 * x - 1);

	fit_a = lerp(fit_min_scat_a, fit_a, lambda_scat);
	fit_b = lerp(fit_min_scat_b, fit_min_scat_b, lambda_scat);

	return lerp(fit_b, fit_a, y * y);
}

float3 IntegrateDiffuseScattering(float3 NdotL, float radius, float3 shapeParam)
{
    return float3(IntegrateDiffuseScattering(NdotL.x, radius, shapeParam.x),
                  IntegrateDiffuseScattering(NdotL.y, radius, shapeParam.y),
                  IntegrateDiffuseScattering(NdotL.z, radius, shapeParam.z));
}
