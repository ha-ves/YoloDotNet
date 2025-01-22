using System.Runtime.Versioning;
using YoloDotNet.Enums;

namespace YoloDotNet.Core
{
    public abstract class YoloCoreGPUBase : YoloCore
    {
        protected YoloCoreGPUBase(string onnxModel, bool useCuda, bool allocateGpuMemory, int gpuId) : base(onnxModel, useCuda, allocateGpuMemory, gpuId)
        {
        }
    }

    public class YoloProviderFactory
    {
        private static Type? type;

        public static YoloCore GetCoreProvider(string onnxModel, bool cuda, bool primeGpu, int gpuId)
        {
            if(type is null)
                throw new NotImplementedException("No GPU Implementation of Yolo Pre & Post Processing available. Add some with the other packages.");

            return (YoloCore)Activator.CreateInstance(type, onnxModel, cuda, primeGpu, gpuId)!;
        }

        public static void RegisterCore<T>()
            where T : YoloCoreGPUBase
        {
            if (!Attribute.IsDefined(typeof(T), typeof(SupportedOSPlatformAttribute)))
                throw new NotImplementedException($"{typeof(YoloCoreGPUBase)} Derived class " +
                    $"must have {nameof(SupportedOSPlatformAttribute)} annotation.");

            var support = (SupportedOSPlatformAttribute)Attribute.GetCustomAttribute(typeof(T), typeof(SupportedOSPlatformAttribute))!;

            if (!OperatingSystem.IsOSPlatform(support.PlatformName))
                throw new NotSupportedException($"This platform [[[{Environment.OSVersion.VersionString}]]] " +
                    $"is not supported by the provider [{typeof(T)}].");

            type = typeof(T);
        }

        // TODO: better storage facility
        private static Dictionary<ModelVersion, Dictionary<ModelType, Type>?>? moduleTypes;

        // TODO: better chooser mechanism
        public static void RegisterModule<T>(ModelVersion ver, ModelType type) where T : IModule
        {
            if (moduleTypes is null)
                moduleTypes = [];

            if (moduleTypes[ver] is null)
                moduleTypes[ver] = [];

            moduleTypes[ver]![type] = typeof(T);
        }

        public static IModule GetModule(YoloCoreGPUBase core, ModelVersion modelVersion, ModelType modelType)
        {
            if(moduleTypes?[modelVersion]?[modelType] is not null)
                return (IModule)Activator.CreateInstance(moduleTypes![modelVersion]![modelType], core)!;

            throw new NotSupportedException($"Unsupported detection type {modelType} or model version {modelVersion} with {core.GetType()} (No Module).");
        }
    }
}
