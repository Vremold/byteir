//===- module.cc ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/backends/cpu/device/llvm/jit.h"
#include "brt/core/common/common.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include <unordered_set>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace brt {
namespace cpu {
namespace {
constexpr static llvm::StringRef llvmJittedObjbufferSuffix =
    "-jitted-objectbuffer";

inline void consumeLLVMError(llvm::Error err) {
  // use CAPI to avoid involving VTable of llvm::ErrorInfoBase
  LLVMConsumeError(llvm::wrap(std::move(err)));
}

inline llvm::Optional<llvm::OptimizationLevel>
getLLVMOptimizationLevel(int optLevel) {
  switch (optLevel) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  case 3:
    return llvm::OptimizationLevel::O3;
  default:
    // unknown opt level, return None
    return llvm::None;
  }
}

inline void runLLVMDefaultOptimizationPipeline(llvm::Module &M,
                                               llvm::OptimizationLevel optLevel,
                                               llvm::TargetMachine *TM) {
  // Follow llvm tutorial to run default optimization pipeline
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassBuilder PB(TM);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM = optLevel != llvm::OptimizationLevel::O0
                                    ? PB.buildPerModuleDefaultPipeline(optLevel)
                                    : PB.buildO0DefaultPipeline(optLevel);

  MPM.run(M, MAM);
}

llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>>
createRTDyldObjectLinkingLayerWithGDBListner(llvm::orc::ExecutionSession &ES,
                                             const llvm::Triple &TT) {
  auto GetMemMgr = []() {
    return std::make_unique<llvm::SectionMemoryManager>();
  };
  auto Layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      ES, std::move(GetMemMgr));

  Layer->setProcessAllSections(true);
  Layer->registerJITEventListener(
      *llvm::JITEventListener::createGDBRegistrationListener());

  return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(Layer));
}

inline std::string makePackedFunctionName(llvm::StringRef name) {
  return "_packed_" + name.str();
}

void packFunctionArguments(llvm::Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  llvm::DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto *newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc =
        llvm::cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto &indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
      llvm::Value *argPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, argIndex);
      llvm::Value *argPtr =
          builder.CreateLoad(builder.getInt8PtrTy(), argPtrPtr);
      llvm::Type *argTy = indexedArg.value().getType();
      argPtr = builder.CreateBitCast(argPtr, argTy->getPointerTo());
      llvm::Value *arg = builder.CreateLoad(argTy, argPtr);
      args.push_back(arg);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, retIndex);
      llvm::Value *retPtr =
          builder.CreateLoad(builder.getInt8PtrTy(), retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}
} // namespace
// thin wrapper around LLJIT but use brt::Status as return code
class LLVMJITImpl {
public:
  struct Options {
    llvm::Optional<llvm::OptimizationLevel>
        optLevel; // None for no optimization
    bool debug;   // whether to save debug infomations
    Options(int optLevel_, bool debug_)
        : optLevel(getLLVMOptimizationLevel(optLevel_)), debug(debug_) {}
  };

  LLVMJITImpl(Options options);

  common::Status LoadTSM(llvm::orc::ThreadSafeModule &&tsm);

  common::Status ParseIRFile(const std::string &path);

  common::Status Lookup(const std::string &symbolName, void **symbol);
  common::Status LookupPacked(const std::string &symbolName, void **symbol) {
    return Lookup(makePackedFunctionName(symbolName), symbol);
  }

  common::Status RegisterSymbol(const std::string &symbol, void *addr);

  common::Status PrintOptimizedModule(const std::string &identifier,
                                      std::ostream &os);

  common::Status DumpObject(const std::string &identifier, std::ostream &os);

private:
  void InitRuntimeLibcalls();

  struct DebugInfo {
    // TODO?: multi-threads
    std::unordered_map<std::string, std::string> identifier2mod;
    std::unordered_map<std::string, std::string> identifier2obj;
  };

  Options options;
  std::unique_ptr<llvm::orc::LLJIT> jit;
  llvm::DenseSet<llvm::orc::SymbolStringPtr> rt_libcalls;
  DebugInfo dbgInfo;
};

LLVMJITImpl::LLVMJITImpl(Options opt) : options(opt) {
  if (opt.debug) {
    jit = llvm::cantFail(llvm::orc::LLJITBuilder()
                             .setObjectLinkingLayerCreator(
                                 createRTDyldObjectLinkingLayerWithGDBListner)
                             .create());
  } else {
    jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());
  }

  // Make sure that our process symbols are visible to JIT'd code.
  jit->getMainJITDylib().addGenerator(llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix(),
          [this](const llvm::orc::SymbolStringPtr &name) {
            return rt_libcalls.count(name);
          })));

  jit->getIRTransformLayer().setTransform(
      [this](llvm::orc::ThreadSafeModule TSM,
             const llvm::orc::MaterializationResponsibility &)
          -> llvm::Expected<llvm::orc::ThreadSafeModule> {
        TSM.withModuleDo([&](llvm::Module &M) {
          if (options.optLevel.has_value()) {
            auto JTMB = llvm::cantFail(
                llvm::orc::JITTargetMachineBuilder::detectHost());
            auto TM = llvm::cantFail(JTMB.createTargetMachine());
            runLLVMDefaultOptimizationPipeline(M, options.optLevel.value(),
                                               TM.get());
          }
        });
        return std::move(TSM);
      });

  jit->getIRCompileLayer().setNotifyCompiled(
      [this](llvm::orc::MaterializationResponsibility &,
             llvm::orc::ThreadSafeModule TSM) {
        TSM.withModuleDo([&](llvm::Module &M) {
          if (options.debug) {
            std::string &s = dbgInfo.identifier2mod[M.getModuleIdentifier()];
            llvm::raw_string_ostream ss(s);
            M.print(ss, nullptr);
          }
        });
      });

  jit->getObjTransformLayer().setTransform(
      [this](std::unique_ptr<llvm::MemoryBuffer> MB) {
        if (options.debug) {
          dbgInfo.identifier2obj[MB->getBufferIdentifier().str()] =
              MB->getBuffer().str();
        }
        return std::move(MB);
      });

  InitRuntimeLibcalls();
}

common::Status LLVMJITImpl::LoadTSM(llvm::orc::ThreadSafeModule &&tsm) {
  tsm.withModuleDo([&](llvm::Module &M) { packFunctionArguments(&M); });
  auto err = jit->addIRModule(std::move(tsm));
  if (err) {
    consumeLLVMError(std::move(err));
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Load TSM failed");
  }
  return common::Status::OK();
}

common::Status LLVMJITImpl::ParseIRFile(const std::string &path) {
  auto ctx = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic err;
  auto mod = llvm::parseIRFile(path, err, *ctx);
  if (!mod) {
    // TODO: handle err message
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Parse LLVM module failed");
  }
  mod->setModuleIdentifier(path);

  return LoadTSM({std::move(mod), std::move(ctx)});
}

common::Status LLVMJITImpl::Lookup(const std::string &symbolName,
                                   void **symbol) {
  auto expectedSymbol = jit->lookup(symbolName);
  if (!expectedSymbol) {
    consumeLLVMError(expectedSymbol.takeError());
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Unexpected symbol in llvm module");
  }
  if (symbol) {
    *symbol = expectedSymbol->toPtr<void *>();
  }
  return common::Status::OK();
}

common::Status LLVMJITImpl::RegisterSymbol(const std::string &symbol,
                                           void *addr) {
  auto &mainJitDylib = jit->getMainJITDylib();
  auto interner = llvm::orc::MangleAndInterner(
      mainJitDylib.getExecutionSession(), jit->getDataLayout());
  llvm::orc::SymbolMap symbolMap;
  symbolMap[interner(symbol)] = llvm::JITEvaluatedSymbol::fromPointer(addr);
  auto err = mainJitDylib.define(llvm::orc::absoluteSymbols(symbolMap));
  if (err) {
    consumeLLVMError(std::move(err));
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Failed to register symbol");
  }
  return common::Status::OK();
}

common::Status LLVMJITImpl::PrintOptimizedModule(const std::string &identifier,
                                                 std::ostream &os) {
  auto &&dbgMap = dbgInfo.identifier2mod;
  auto &&iter = dbgMap.find(identifier);
  if (iter == dbgMap.end()) {
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Cannot find optimized llvm module");
  }
  os << iter->second;
  return common::Status::OK();
}

common::Status LLVMJITImpl::DumpObject(const std::string &identifier,
                                       std::ostream &os) {
  auto &&dbgMap = dbgInfo.identifier2obj;
  auto &&iter = dbgMap.find(identifier + llvmJittedObjbufferSuffix.str());
  if (iter == dbgMap.end()) {
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Cannot find object");
  }
  os << iter->second;
  return common::Status::OK();
}

void LLVMJITImpl::InitRuntimeLibcalls() {
  llvm::orc::MangleAndInterner mangle(jit->getExecutionSession(),
                                      jit->getDataLayout());
#define HANDLE_LIBCALL(code, name)                                             \
  if (name)                                                                    \
    rt_libcalls.insert(mangle(name ? name : ""));
#include "llvm/IR/RuntimeLibcalls.def"
#undef HANDLE_LIBCALL
}

#if BRT_LLJIT_DEBUG
LLVMJIT::LLVMJIT()
    : impl{new LLVMJITImpl({/* optLevel */ 0, /* debug */ true})} {}
#else
LLVMJIT::LLVMJIT()
    : impl{new LLVMJITImpl({/* optLevel */ 3, /* debug */ false})} {}
#endif

LLVMJIT::~LLVMJIT() = default;

common::Status LLVMJIT::LoadFromFile(const std::string &path) {
  return impl->ParseIRFile(path);
}

common::Status LLVMJIT::LoadFromBuffer(void *buf) {
  auto &&tsm = *reinterpret_cast<llvm::orc::ThreadSafeModule *>(buf);
  return impl->LoadTSM(std::move(tsm));
}

common::Status LLVMJIT::Lookup(const std::string &symbolName, void **symbol) {
  return impl->Lookup(symbolName, symbol);
}

common::Status LLVMJIT::LookupPacked(const std::string &symbolName,
                                     void **symbol) {
  return impl->LookupPacked(symbolName, symbol);
}

common::Status LLVMJIT::RegisterSymbol(const std::string &symbol_name,
                                       void *symbol) {
  return impl->RegisterSymbol(symbol_name, symbol);
}

common::Status LLVMJIT::PrintOptimizedModule(const std::string &identifier,
                                             std::ostream &os) {
  return impl->PrintOptimizedModule(identifier, os);
}

common::Status LLVMJIT::DumpObject(const std::string &identifier,
                                   std::ostream &os) {
  return impl->DumpObject(identifier, os);
}

LLVMJIT *LLVMJIT::Instance() {
  static auto initLLVM = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  static_cast<void>(initLLVM);

  static LLVMJIT inst;
  return &inst;
}
} // namespace cpu
} // namespace brt